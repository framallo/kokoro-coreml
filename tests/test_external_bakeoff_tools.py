import csv
import hashlib
import json
from pathlib import Path

from scripts.external_bakeoff.create_listening_review import _write_decisions_csv
from scripts.external_bakeoff.ingest_ios_runner_result import _ingest_records
from scripts.external_bakeoff.run_laishere_kokoro_coreml import (
    DEFAULT_COMPUTE_UNITS as LAISHERE_DEFAULT_COMPUTE_UNITS,
    _compute_units as laishere_compute_units,
)
from scripts.external_bakeoff.schema import (
    RUNTIME_BUCKETS,
    result_file_payload,
    result_record,
    validate_result_payload,
)
from scripts.external_bakeoff.summarize_competitive_frontier import (
    load_records as load_frontier_records,
    render_markdown as render_frontier_markdown,
    summarize_frontier,
)
from scripts.external_bakeoff.summarize_frontier_freshness import (
    render_markdown as render_frontier_freshness_markdown,
    summarize_freshness,
)
from scripts.external_bakeoff.summarize_irvine_next_targets import (
    render_markdown as render_irvine_next_targets_markdown,
    summarize_targets as summarize_irvine_next_targets,
)
from scripts.external_bakeoff.summarize_stage_gap_decomposition import (
    render_markdown as render_stage_gap_markdown,
    summarize_config_record,
    summarize_laishere_record,
    summarize_stage_gaps,
)
from scripts.external_bakeoff.validate_listening_decisions import validate_rows
from scripts.external_bakeoff.verify_external_bakeoff_completion import (
    _check_iphone,
    _check_mac_primary,
    _check_preflight,
    _load_records,
)
from scripts.compare_coreml_metadata import (
    render_markdown as render_coreml_metadata_markdown,
    summarize_metadata as summarize_coreml_metadata,
)
from scripts.analyze_waveform_alignment import analyze_pair
from scripts.create_f0_source_listening_pack import _write_decisions_csv as _write_f0_decisions_csv
from scripts.probe_generator_cos_snake import _trim_or_pad_last_dim
from scripts.summarize_f0_source_candidates import (
    collect_rows as collect_f0_candidate_rows,
    load_decisions as load_f0_candidate_decisions,
    render_markdown as render_f0_candidate_markdown,
    summarize_report as summarize_f0_candidate_report,
)
from scripts.summarize_frontier_gap_candidates import (
    infer_bucket as infer_gap_candidate_bucket,
    infer_machine as infer_gap_candidate_machine,
    render_markdown as render_frontier_gap_candidates_markdown,
    summarize_gap_candidates,
)
from scripts.summarize_optimization_candidates import (
    classify as classify_optimization_candidate,
    collect_rows as collect_optimization_candidates,
    render_markdown as render_optimization_candidates_markdown,
)
from scripts.validate_f0_source_listening_decisions import validate_rows as validate_f0_rows


class _FakeComputeUnit:
    ALL = "ALL"
    CPU_AND_GPU = "CPU_AND_GPU"
    CPU_AND_NE = "CPU_AND_NE"
    CPU_ONLY = "CPU_ONLY"


class _FakeCoreMLTools:
    ComputeUnit = _FakeComputeUnit


def _manifest() -> dict:
    inputs = {}
    for index, key in enumerate(RUNTIME_BUCKETS, start=1):
        text = f"runtime text {key}"
        inputs[key] = {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "voice": "af_heart",
            "speed": 1.0,
            "canonical_duration_s": float(index),
            "expected_bucket_s": int(key.rstrip("s")),
        }
    return {"runtime_buckets": list(RUNTIME_BUCKETS), "inputs": inputs}


def _ios_payload(manifest: dict) -> dict:
    records = []
    for key in RUNTIME_BUCKETS:
        item = manifest["inputs"][key]
        duration = item["canonical_duration_s"]
        warm = [0.1, 0.11, 0.12, 0.13, 0.14]
        records.append(
            {
                "input_key": key,
                "text_sha256": item["text_sha256"],
                "voice": item["voice"],
                "canonical_audio_duration_s": duration,
                "expected_bucket_s": item["expected_bucket_s"],
                "cold_wall_time_s": 0.2,
                "warm_wall_times_s": warm,
                "sample_count": int(duration * 24000),
                "sample_rate": 24000,
                "observed_audio_duration_s": duration,
                "rtf_observed": [value / duration for value in warm],
            }
        )
    return {
        "impl": "soniqo-speech-swift-kokoro-ios",
        "framework": "Swift + Core ML",
        "hardware_target": "ANE/Core ML",
        "compute_units": "all",
        "warm_iterations": 5,
        "records": records,
    }


def _config_f_ios_payload(manifest: dict) -> dict:
    records = []
    for key in RUNTIME_BUCKETS:
        item = manifest["inputs"][key]
        duration = item["canonical_duration_s"]
        warm = [0.09, 0.10, 0.11, 0.12, 0.13]
        records.append(
            {
                "input_key": key,
                "text_sha256": item["text_sha256"],
                "voice": item["voice"],
                "canonical_audio_duration_s": duration,
                "expected_bucket_s": item["expected_bucket_s"],
                "post_preflight_cold_wall_time_s": 0.2,
                "warm_wall_times_s": warm,
                "sample_count": int(duration * 24000),
                "observed_audio_duration_s": duration,
                "bucket_used": key,
                "duration_model": f"exact_t{item['expected_bucket_s']}",
                "stage_medians_s": {"generator_coreml": 0.05},
                "raw_warm_stage_timings_s": [{"generator_coreml": 0.05}],
            }
        )
    return {
        "impl": "config-f-reference-ios",
        "framework": "Swift + Core ML",
        "hardware_target": "ANE/Core ML",
        "compute_units": "staged(duration/f0n/generator=cpuAndGPU,decoderPre=cpuAndNeuralEngine)",
        "warm_iterations": 5,
        "records": records,
    }


def test_laishere_compute_unit_policy_defaults_and_aliases():
    assert LAISHERE_DEFAULT_COMPUTE_UNITS == {
        "albert": "cpuAndNeuralEngine",
        "post_albert": "cpuAndNeuralEngine",
        "alignment": "cpuAndNeuralEngine",
        "prosody": "cpuAndNeuralEngine",
        "noise": "all",
        "vocoder": "cpuAndNeuralEngine",
        "tail": "all",
    }
    assert laishere_compute_units(_FakeCoreMLTools, "all") == "ALL"
    assert laishere_compute_units(_FakeCoreMLTools, "CPU_AND_GPU") == "CPU_AND_GPU"
    assert laishere_compute_units(_FakeCoreMLTools, "cpu-and-ne") == "CPU_AND_NE"
    assert laishere_compute_units(_FakeCoreMLTools, "cpuOnly") == "CPU_ONLY"


def test_ingest_ios_runner_result_validates_manifest_and_schema():
    manifest = _manifest()
    records = _ingest_records(
        payload=_ios_payload(manifest),
        manifest=manifest,
        machine_id="iphone-12-pro",
        version="test",
        source_path=Path("ios.json"),
    )

    assert [record["input_key"] for record in records] == list(RUNTIME_BUCKETS)
    assert all(record["machine_id"] == "iphone-12-pro" for record in records)
    assert all(record["status"] == "ok" for record in records)
    assert all(record["output_sha256"] == "" for record in records)
    assert all(record["provenance"]["spotcheck_wav_unavailable"] for record in records)

    validate_result_payload(
        {
            "created_utc": "2026-06-05T00:00:00Z",
            "impl": "soniqo-speech-swift-kokoro-ios",
            "machine_id": "iphone-12-pro",
            "machine": {"machine_id": "iphone-12-pro"},
            "records": records,
            "provenance": {},
        }
    )


def test_ingest_ios_runner_result_accepts_config_f_payload_shape():
    manifest = _manifest()
    records = _ingest_records(
        payload=_config_f_ios_payload(manifest),
        manifest=manifest,
        machine_id="iphone-12-pro",
        version="test",
        source_path=Path("config-f-ios.json"),
    )

    assert [record["input_key"] for record in records] == list(RUNTIME_BUCKETS)
    assert {record["impl"] for record in records} == {"config-f-reference-ios"}
    assert all(record["cold_wall_time_s"] == 0.2 for record in records)
    assert records[0]["provenance"]["duration_model"] == "exact_t3"
    assert records[0]["provenance"]["bucket_used"] == "3s"
    assert records[0]["provenance"]["stage_medians_s"] == {"generator_coreml": 0.05}

    validate_result_payload(
        {
            "created_utc": "2026-06-05T00:00:00Z",
            "impl": "config-f-reference-ios",
            "machine_id": "iphone-12-pro",
            "machine": {"machine_id": "iphone-12-pro"},
            "records": records,
            "provenance": {},
        }
    )


def test_listening_decision_validator_blocks_blank_success_rows():
    summary, errors = validate_rows(
        [
            {
                "machine_id": "m2-studio",
                "input_key": "3s",
                "impl": "config-f-reference",
                "status": "ok",
                "wav_path": "sample.wav",
                "human_decision": "",
                "notes": "",
            },
            {
                "machine_id": "m2-studio",
                "input_key": "3s",
                "impl": "mlx-audio",
                "status": "error",
                "wav_path": "",
                "human_decision": "",
                "notes": "",
            },
        ]
    )

    assert not summary["valid"]
    assert summary["ok_rows"] == 1
    assert summary["error_rows"] == 1
    assert errors == ["m2-studio/3s/config-f-reference: missing human_decision"]


def test_listening_review_regeneration_preserves_human_decisions(tmp_path):
    decisions = tmp_path / "decisions.csv"
    original = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "old.wav",
            "duration_s": "2.800",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "pass",
            "notes": "heard cleanly",
            "expected_text": "old text",
        }
    ]
    regenerated = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "new.wav",
            "duration_s": "2.801",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "",
            "notes": "",
            "expected_text": "new text",
        }
    ]

    _write_decisions_csv(decisions, original)
    _write_decisions_csv(decisions, regenerated)

    row = next(csv.DictReader(decisions.open()))
    assert row["wav_path"] == "new.wav"
    assert row["duration_s"] == "2.801"
    assert row["expected_text"] == "new text"
    assert row["human_decision"] == "pass"
    assert row["notes"] == "heard cleanly"


def test_listening_review_reset_decisions_blanks_human_fields(tmp_path):
    decisions = tmp_path / "decisions.csv"
    rows = [
        {
            "machine_id": "m2-studio",
            "input_key": "3s",
            "impl": "config-f-reference",
            "impl_label": "Config F reference",
            "status": "ok",
            "wav_path": "sample.wav",
            "duration_s": "2.800",
            "waveform_decision": "reference_pass",
            "caveat": "",
            "human_decision": "pass",
            "notes": "heard cleanly",
            "expected_text": "text",
        }
    ]

    _write_decisions_csv(decisions, rows)
    reset_rows = [dict(rows[0], human_decision="", notes="")]
    _write_decisions_csv(decisions, reset_rows, preserve_existing=False)

    row = next(csv.DictReader(decisions.open()))
    assert row["human_decision"] == ""
    assert row["notes"] == ""


def test_f0_source_decision_validator_blocks_blank_rows():
    summary, errors = validate_f0_rows(
        [
            {
                "label": "candidate",
                "waveform_gate_decision": "needs_listening",
                "candidate_wav": "candidate.wav",
                "review": "review.md",
                "source_report": "report.json",
                "human_decision": "",
                "notes": "",
            }
        ]
    )

    assert not summary["valid"]
    assert summary["rows"] == 1
    assert errors == ["candidate: missing human_decision"]


def test_f0_source_decision_validator_requires_caveat_notes():
    summary, errors = validate_f0_rows(
        [
            {
                "label": "candidate",
                "waveform_gate_decision": "needs_listening",
                "candidate_wav": "candidate.wav",
                "review": "review.md",
                "source_report": "report.json",
                "human_decision": "caveat",
                "notes": "",
            }
        ]
    )

    assert not summary["valid"]
    assert errors == ["candidate: human_decision=caveat requires notes"]


def test_f0_source_decision_csv_preserves_human_fields(tmp_path):
    decisions = tmp_path / "f0_decisions.csv"
    original = [
        {
            "label": "candidate",
            "waveform_gate_decision": "needs_listening",
            "candidate_wav": "old.wav",
            "review": "old.md",
            "source_report": "old.json",
            "human_decision": "pass",
            "notes": "acceptable source character",
        }
    ]
    regenerated = [
        {
            "label": "candidate",
            "waveform_gate_decision": "needs_listening",
            "candidate_wav": "new.wav",
            "review": "new.md",
            "source_report": "new.json",
            "human_decision": "",
            "notes": "",
        }
    ]

    _write_f0_decisions_csv(decisions, original)
    _write_f0_decisions_csv(decisions, regenerated)

    row = next(csv.DictReader(decisions.open()))
    assert row["candidate_wav"] == "new.wav"
    assert row["review"] == "new.md"
    assert row["source_report"] == "new.json"
    assert row["human_decision"] == "pass"
    assert row["notes"] == "acceptable source character"


def test_f0_source_candidate_summary_ranks_warm_speedups_and_decisions(tmp_path):
    report_path = tmp_path / "outputs" / "f0_noise_exact_shape" / "3s_candidate" / "report.json"
    report_path.parent.mkdir(parents=True)
    source_report = "outputs/f0_noise_exact_shape/3s_candidate/report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": source_report,
                "tensor_dump": "outputs/external_bakeoff/tensor_dumps/3s",
                "passes": False,
                "export": {
                    "natural_asr": True,
                    "deployment_target": "iOS18",
                    "native_instance_norm": False,
                    "palettize_body": False,
                    "source_mode": "cos_rsqrt",
                    "phase_mode": "atan2",
                },
                "benchmark": {
                    "warm_predict_median_ms": {
                        "baseline_total": 100.0,
                        "candidate_total": 75.0,
                    },
                    "metrics": {
                        "candidate_vs_baseline_trimmed": {
                            "correlation": 0.95,
                            "snr_db": 10.5,
                            "max_abs_error": 0.2,
                        }
                    },
                },
            }
        )
    )
    slower_path = tmp_path / "outputs" / "f0_noise_exact_shape" / "7s_candidate" / "report.json"
    slower_path.parent.mkdir(parents=True)
    slower_path.write_text(
        json.dumps(
            {
                "report": "outputs/f0_noise_exact_shape/7s_candidate/report.json",
                "tensor_dump": "outputs/external_bakeoff/tensor_dumps/7s",
                "passes": False,
                "benchmark": {
                    "warm_predict_median_ms": {
                        "baseline_total": 100.0,
                        "candidate_total": 90.0,
                    },
                    "metrics": {"candidate_vs_baseline_trimmed": {}},
                },
            }
        )
    )
    decisions_path = tmp_path / "decisions.csv"
    decisions_path.write_text(
        "\n".join(
            [
                "label,waveform_gate_decision,candidate_wav,review,source_report,human_decision,notes",
                "3s_candidate,needs_listening,candidate.wav,review.md,"
                + source_report
                + ",caveat,acceptable but brighter",
                "",
            ]
        )
    )

    decisions = load_f0_candidate_decisions([decisions_path])
    row = summarize_f0_candidate_report(report_path, decisions)
    assert row
    assert row["label"] == "3s_candidate"
    assert row["bucket"] == "3s"
    assert row["speedup_pct"] == 25.0
    assert row["corr"] == 0.95
    assert row["snr_db"] == 10.5
    assert row["passes_strict_gate"] is False
    assert row["natural_asr"] is True
    assert row["human_decision"] == "caveat"
    assert row["decision_notes"] == "acceptable but brighter"

    rows = collect_f0_candidate_rows([str(tmp_path / "outputs/f0_noise_exact_shape/**/report.json")], decisions)
    assert [item["label"] for item in rows] == ["3s_candidate", "7s_candidate"]

    markdown = render_f0_candidate_markdown(rows)
    assert "| 1 | m2-studio | 3s | `3s_candidate` | 25.0% |" in markdown
    assert "Strict waveform failures are not production approvals" in markdown


def test_competitive_frontier_filters_short_outputs_and_marks_losses(tmp_path):
    results = tmp_path / "results.json"
    payload = {
        "created_utc": "2026-06-06T00:00:00Z",
        "impl": "mixed",
        "machine_id": "mixed",
        "machine": {"machine_id": "mixed"},
        "records": [
            {
                "impl": "config-f-reference",
                "machine_id": "m2-air",
                "input_key": "3s",
                "status": "ok",
                "warm_wall_times_s": [0.150, 0.148, 0.149],
                "canonical_audio_duration_s": 2.8,
                "observed_audio_duration_s": 2.8,
            },
            {
                "impl": "laishere-kokoro-coreml",
                "machine_id": "m2-air",
                "input_key": "3s",
                "status": "ok",
                "warm_wall_times_s": [0.142, 0.144, 0.143],
                "canonical_audio_duration_s": 2.8,
                "observed_audio_duration_s": 2.775,
            },
            {
                "impl": "soniqo-speech-swift-kokoro",
                "machine_id": "m2-air",
                "input_key": "7s",
                "status": "ok",
                "warm_wall_times_s": [0.070, 0.071, 0.069],
                "canonical_audio_duration_s": 6.75,
                "observed_audio_duration_s": 5.0,
            },
            {
                "impl": "soniqo-speech-swift-kokoro-ios",
                "machine_id": "iphone-12-pro",
                "input_key": "3s",
                "status": "ok",
                "warm_wall_times_s": [0.830, 0.831, 0.829],
                "canonical_audio_duration_s": 2.8,
                "observed_audio_duration_s": 2.7,
            },
            {
                "impl": "config-f-reference-ios",
                "machine_id": "iphone-12-pro",
                "input_key": "7s",
                "status": "ok",
                "warm_wall_times_s": [0.700, 0.701, 0.699],
                "canonical_audio_duration_s": 6.75,
                "observed_audio_duration_s": 6.75,
            },
            {
                "impl": "soniqo-speech-swift-kokoro-ios",
                "machine_id": "iphone-12-pro",
                "input_key": "7s",
                "status": "ok",
                "warm_wall_times_s": [0.830, 0.831, 0.829],
                "canonical_audio_duration_s": 6.75,
                "observed_audio_duration_s": 5.0,
            },
        ],
        "provenance": {},
    }
    results.write_text(json.dumps(payload))

    rows = load_frontier_records([results], min_duration_ratio=0.95)
    summary = summarize_frontier(rows)
    m2_air_3s = next(
        cell
        for cell in summary["cells"]
        if cell["machine_id"] == "m2-air" and cell["input_key"] == "3s"
    )
    m2_air_7s = next(
        cell
        for cell in summary["cells"]
        if cell["machine_id"] == "m2-air" and cell["input_key"] == "7s"
    )
    iphone_3s = next(
        cell
        for cell in summary["cells"]
        if cell["machine_id"] == "iphone-12-pro" and cell["input_key"] == "3s"
    )
    iphone_7s = next(
        cell
        for cell in summary["cells"]
        if cell["machine_id"] == "iphone-12-pro" and cell["input_key"] == "7s"
    )

    assert m2_air_3s["outcome"] == "config-f-loses"
    assert m2_air_3s["best_impl"] == "laishere-kokoro-coreml"
    assert round(m2_air_3s["gap_pct"], 1) == 4.2
    assert m2_air_7s["outcome"] == "no-full-duration-result"
    assert m2_air_7s["best_impl"] is None
    assert iphone_3s["outcome"] == "config-f-missing"
    assert iphone_3s["best_impl"] == "soniqo-speech-swift-kokoro-ios"
    assert iphone_7s["outcome"] == "config-f-wins"
    assert iphone_7s["best_impl"] == "config-f-reference-ios"
    assert summary["absolute_fastest_verified"] is False

    markdown = render_frontier_markdown(summary, rows, min_duration_ratio=0.95)
    assert "Config F needs 4.2%" in markdown
    assert "Excluded Short Outputs" in markdown
    assert "| m2-air | 7s | Soniqo | 70.0 ms | 0.741 |" in markdown


def test_frontier_freshness_flags_stage_profile_ties():
    frontier = {
        "summary": {
            "config_f_losses": [
                {
                    "machine_id": "m2-air",
                    "input_key": "3s",
                    "best_impl_label": "laishere",
                    "best_warm_median_ms": 142.0,
                    "config_f_warm_median_ms": 148.0,
                },
                {
                    "machine_id": "irvine-m1",
                    "input_key": "3s",
                    "best_impl_label": "laishere",
                    "best_warm_median_ms": 176.3,
                    "config_f_warm_median_ms": 233.6,
                },
            ]
        }
    }
    config_payloads = {
        "m2-air": {
            "records": [
                {"input_key": "3s", "warm_wall_times_s": [0.148, 0.149, 0.147]},
            ]
        },
        "irvine-m1": {
            "records": [
                {"input_key": "3s", "warm_wall_times_s": [0.233, 0.234, 0.235]},
            ]
        },
    }
    stage_profiles = {
        "m2-air": {
            "records": [
                {"input_key": "3s", "warm_median_s": {"total_s": 0.153}},
            ]
        },
        "irvine-m1": {
            "records": [
                {"input_key": "3s", "warm_median_s": {"total_s": 0.195}},
            ]
        },
    }

    summary = summarize_freshness(frontier, config_payloads, stage_profiles, tie_threshold_pct=2.0)

    assert summary["frontier_loss_count"] == 2
    assert summary["stale_or_tie_loss_count"] == 1
    assert summary["real_profile_loss_count"] == 1
    m2_air = summary["loss_rows"][0]
    irvine = summary["loss_rows"][1]
    assert m2_air["profile_outcome"] == "profile-config-f-wins"
    assert m2_air["frontier_loss_looks_stale_or_tie"] is True
    assert irvine["profile_outcome"] == "profile-config-f-loses"
    assert irvine["frontier_loss_looks_stale_or_tie"] is False

    markdown = render_frontier_freshness_markdown(summary)
    assert "frontier loss likely stale" in markdown
    assert "real remaining loss" in markdown


def test_optimization_candidate_summary_separates_quality_from_speed(tmp_path):
    safe = tmp_path / "outputs" / "generator_cos_snake" / "safe" / "report.json"
    safe.parent.mkdir(parents=True)
    safe.write_text(
        json.dumps(
            {
                "passes": True,
                "report": "outputs/generator_cos_snake/safe/report.json",
                "benchmark": {
                    "warm_predict_median_ms": {
                        "fused": 100.0,
                        "candidate": 98.5,
                    },
                    "metrics": {
                        "candidate_vs_fused_trimmed": {
                            "correlation": 0.99999,
                            "snr_db": 50.0,
                            "max_abs_error": 0.001,
                        }
                    },
                },
            }
        )
    )
    unsafe = tmp_path / "outputs" / "f0_noise_exact_shape" / "fast_fail" / "report.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text(
        json.dumps(
            {
                "passes": False,
                "speedup_vs_baseline_pct": 12.0,
                "benchmark": {
                    "warm_predict_median_ms": {
                        "baseline_total": 100.0,
                        "candidate_total": 88.0,
                    },
                    "metrics": {
                        "candidate_vs_baseline_trimmed": {
                            "correlation": 0.80,
                            "snr_db": 5.0,
                            "max_abs_error": 0.5,
                        }
                    },
                },
            }
        )
    )

    rows = collect_optimization_candidates([str(tmp_path / "outputs/**/report.json")])
    assert [row["label"] for row in rows] == ["fast_fail", "safe"]
    assert classify_optimization_candidate(rows[0], material_speedup_pct=3.0) == "speed-positive quality fail"
    assert classify_optimization_candidate(rows[1], material_speedup_pct=3.0) == "quality-safe noise-sized speedup"

    markdown = render_optimization_candidates_markdown(rows, material_speedup_pct=3.0, top=10)
    assert "Quality-safe material candidates: `0`." in markdown
    assert "speed-positive quality fail" in markdown
    assert "quality-safe noise-sized speedup" in markdown


def test_frontier_gap_candidate_infers_bucket_and_machine():
    irvine = {
        "label": "10s_padded_cos_resblock_phase_acos_cos_rsqrt",
        "path": "outputs/f0_noise_exact_shape/remote_reports/report_f0_noise_phase_acos_10s_irvine.json",
    }
    m2air = {
        "label": "3s_broadcast_adain_native_in_ios17",
        "path": "outputs/generator_cos_snake/3s_broadcast_adain_native_in_ios17/report_m2air_ios17_native_broadcast_cos.json",
    }

    assert infer_gap_candidate_bucket(irvine) == "10s"
    assert infer_gap_candidate_machine(irvine) == "irvine-m1"
    assert infer_gap_candidate_bucket(m2air) == "3s"
    assert infer_gap_candidate_machine(m2air) == "m2-air"


def test_frontier_gap_candidate_estimates_exact_machine_closure():
    frontier = {
        "config_f_losses": [
            {
                "machine_id": "irvine-m1",
                "input_key": "10s",
                "best_impl_label": "laishere",
                "best_warm_median_ms": 590.0,
                "config_f_warm_median_ms": 685.0,
            },
            {
                "machine_id": "m2-air",
                "input_key": "3s",
                "best_impl_label": "laishere",
                "best_warm_median_ms": 142.0,
                "config_f_warm_median_ms": 148.0,
            },
        ]
    }
    rows = [
        {
            "family": "f0_noise_exact_shape",
            "label": "10s_padded_cos_resblock_phase_acos_cos_rsqrt",
            "path": "outputs/f0_noise_exact_shape/remote_reports/report_f0_noise_phase_acos_10s_irvine.json",
            "passes": False,
            "speedup_pct": 10.0,
            "baseline_ms": 560.0,
            "candidate_ms": 500.0,
            "corr": 0.96,
            "snr_db": 12.0,
        },
        {
            "family": "generator_cos_snake",
            "label": "3s_broadcast_adain_native_in_ios17",
            "path": "outputs/generator_cos_snake/3s_broadcast_adain_native_in_ios17/report_m2air_ios17_native_broadcast_cos.json",
            "passes": True,
            "speedup_pct": 4.8,
            "baseline_ms": 100.0,
            "candidate_ms": 92.0,
            "corr": 0.99999,
            "snr_db": 50.0,
        },
    ]

    summary = summarize_gap_candidates(frontier, rows, top_per_cell=5)
    irvine = next(cell for cell in summary["loss_cells"] if cell["machine_id"] == "irvine-m1")
    m2air = next(cell for cell in summary["loss_cells"] if cell["machine_id"] == "m2-air")

    assert irvine["top_candidates"][0]["estimated_config_f_ms"] == 625.0
    assert irvine["top_candidates"][0]["would_close_gap"] is False
    assert m2air["top_candidates"][0]["estimated_config_f_ms"] == 140.0
    assert m2air["top_candidates"][0]["would_close_gap"] is True
    assert summary["strict_pass_closers"] == 1
    assert summary["quality_fail_closers"] == 0

    markdown = render_frontier_gap_candidates_markdown(summary)
    assert "Strict-pass candidates that would close a loss: `1`." in markdown
    assert "`10s_padded_cos_resblock_phase_acos_cos_rsqrt` -> 625.0 ms (short)" in markdown


def test_stage_gap_decomposition_uses_warmed_stage_medians(tmp_path):
    config_record = {
        "input_key": "3s",
        "warm_wall_times_s": [0.100, 0.120, 0.110],
        "provenance": {
            "raw_warm_results": [
                {
                    "wall_time_s": 0.100,
                    "t_duration_coreml_s": 0.010,
                    "t_f0ntrain_coreml_s": 0.020,
                    "t_decoder_pre_coreml_s": 0.003,
                    "t_coreml_predict_s": 0.050,
                    "t_hnsf_swift_s": 0.006,
                    "t_matrix_ops_s": 0.001,
                    "t_padding_s": 0.001,
                    "t_trim_s": 0.001,
                    "t_alignment_s": 0.0,
                },
                {
                    "wall_time_s": 0.120,
                    "t_duration_coreml_s": 0.012,
                    "t_f0ntrain_coreml_s": 0.022,
                    "t_decoder_pre_coreml_s": 0.004,
                    "t_coreml_predict_s": 0.060,
                    "t_hnsf_swift_s": 0.007,
                    "t_matrix_ops_s": 0.001,
                    "t_padding_s": 0.001,
                    "t_trim_s": 0.001,
                    "t_alignment_s": 0.0,
                },
                {
                    "wall_time_s": 0.110,
                    "t_duration_coreml_s": 0.011,
                    "t_f0ntrain_coreml_s": 0.021,
                    "t_decoder_pre_coreml_s": 0.005,
                    "t_coreml_predict_s": 0.055,
                    "t_hnsf_swift_s": 0.008,
                    "t_matrix_ops_s": 0.001,
                    "t_padding_s": 0.001,
                    "t_trim_s": 0.001,
                    "t_alignment_s": 0.0,
                },
            ]
        },
    }
    config = summarize_config_record(config_record)
    assert config["total_s"] == 0.110
    assert config["generator_s"] == 0.055
    assert round(config["non_generator_s"], 3) == 0.055
    assert round(config["host_other_s"], 3) == 0.009

    laishere = summarize_laishere_record(
        {
            "prepare_wall_time_s": 0.004,
            "warm_median_s": {
                "total_s": 0.090,
                "albert_s": 0.010,
                "post_albert_s": 0.011,
                "alignment_s": 0.001,
                "prosody_s": 0.002,
                "noise_s": 0.020,
                "vocoder_s": 0.040,
                "tail_s": 0.003,
                "python_overhead_s": 0.0,
            },
        }
    )
    assert laishere["noise_vocoder_tail_s"] == 0.063
    assert laishere["other_plus_prepare_s"] == 0.028


def test_stage_gap_decomposition_renders_loss_rows(tmp_path):
    frontier = {
        "summary": {
            "config_f_losses": [
                {
                    "machine_id": "irvine-m1",
                    "input_key": "3s",
                    "best_impl_label": "laishere",
                    "best_warm_median_ms": 90.0,
                    "config_f_warm_median_ms": 110.0,
                }
            ]
        }
    }
    config_payload = {
        "records": [
            {
                "input_key": "3s",
                "status": "ok",
                "warm_wall_times_s": [0.110],
                "provenance": {
                    "raw_warm_results": [
                        {
                            "wall_time_s": 0.110,
                            "t_duration_coreml_s": 0.011,
                            "t_f0ntrain_coreml_s": 0.021,
                            "t_decoder_pre_coreml_s": 0.005,
                            "t_coreml_predict_s": 0.055,
                            "t_hnsf_swift_s": 0.008,
                            "t_matrix_ops_s": 0.001,
                            "t_padding_s": 0.001,
                            "t_trim_s": 0.001,
                            "t_alignment_s": 0.0,
                        }
                    ]
                },
            }
        ]
    }
    laishere_payload = {
        "records": [
            {
                "input_key": "3s",
                "status": "ok",
                "prepare_wall_time_s": 0.004,
                "warm_median_s": {
                    "total_s": 0.090,
                    "albert_s": 0.010,
                    "post_albert_s": 0.011,
                    "alignment_s": 0.001,
                    "prosody_s": 0.002,
                    "noise_s": 0.020,
                    "vocoder_s": 0.040,
                    "tail_s": 0.003,
                },
            }
        ]
    }
    results = tmp_path / "results"
    placement = results / "placement"
    placement.mkdir(parents=True)
    (results / "results_config_f_reference_irvine-m1_vector_noise_batch.json").write_text(
        json.dumps(config_payload)
    )
    (placement / "results_laishere_stage_profile_irvine-m1.json").write_text(
        json.dumps(laishere_payload)
    )

    summary = summarize_stage_gaps(frontier, results, placement)
    row = summary["rows"][0]
    assert round(row["total_gap_s"], 3) == 0.020
    assert round(row["config_generator_minus_laishere_nvt_s"], 3) == -0.008
    assert round(row["config_nongenerator_minus_laishere_other_prepare_s"], 3) == 0.027

    markdown = render_stage_gap_markdown(summary)
    assert "Loss rows analyzed: `1`." in markdown
    assert "| irvine-m1 | 3s | 110.0 ms | 90.0 ms | 20.0 ms |" in markdown


def test_irvine_next_targets_keeps_only_real_irvine_losses():
    freshness = {
        "loss_rows": [
            {
                "machine_id": "m2-air",
                "input_key": "3s",
                "profile_outcome": "profile-tie",
                "profile_config_f_ms": 148.0,
                "profile_laishere_ms": 149.0,
                "profile_gap_ms": -1.0,
                "profile_gap_pct": -0.7,
            },
            {
                "machine_id": "irvine-m1",
                "input_key": "3s",
                "profile_outcome": "profile-config-f-loses",
                "profile_config_f_ms": 233.6,
                "profile_laishere_ms": 195.0,
                "profile_gap_ms": 38.6,
                "profile_gap_pct": 19.8,
            },
            {
                "machine_id": "irvine-m1",
                "input_key": "7s",
                "profile_outcome": "profile-config-f-loses",
                "profile_config_f_ms": 492.7,
                "profile_laishere_ms": 444.2,
                "profile_gap_ms": 48.5,
                "profile_gap_pct": 10.9,
            },
        ]
    }
    stage_gaps = {
        "rows": [
            {
                "machine_id": "irvine-m1",
                "input_key": "3s",
                "config_generator_minus_laishere_nvt_s": 0.022,
                "config_nongenerator_minus_laishere_other_prepare_s": 0.013,
            },
            {
                "machine_id": "irvine-m1",
                "input_key": "7s",
                "config_generator_minus_laishere_nvt_s": 0.043,
                "config_nongenerator_minus_laishere_other_prepare_s": 0.004,
            },
        ]
    }
    gap_candidates = {
        "loss_cells": [
            {
                "machine_id": "irvine-m1",
                "input_key": "3s",
                "strict_pass_closers": 0,
                "quality_fail_closers": 0,
                "top_candidates": [
                    {
                        "quality_status": "quality-fail",
                        "label": "3s_fast_fail",
                        "family": "f0_noise_exact_shape",
                        "delta_ms": 18.0,
                        "estimated_margin_ms": -20.0,
                    },
                    {
                        "quality_status": "strict-pass",
                        "label": "3s_strict_small",
                        "family": "generator_har_input_trim",
                        "delta_ms": 0.7,
                        "estimated_margin_ms": -37.9,
                    },
                ],
            },
            {
                "machine_id": "irvine-m1",
                "input_key": "7s",
                "strict_pass_closers": 0,
                "quality_fail_closers": 0,
                "top_candidates": [],
            },
        ]
    }

    summary = summarize_irvine_next_targets(freshness, stage_gaps, gap_candidates)
    assert summary["real_loss_count"] == 2
    assert summary["strict_pass_closers"] == 0
    assert summary["quality_fail_closers"] == 0
    assert summary["rows"][0]["target_class"] == "source/body primary; upstream/runtime material"
    assert summary["rows"][1]["target_class"] == "source/body dominates"
    assert summary["rows"][0]["best_strict_candidate"]["label"] == "3s_strict_small"
    assert summary["rows"][0]["best_quality_fail_signal"]["label"] == "3s_fast_fail"

    markdown = render_irvine_next_targets_markdown(summary)
    assert "Real Irvine loss rows: `2`." in markdown
    assert "source/body primary; upstream/runtime material" in markdown
    assert "Do not promote fresh Irvine timing" in markdown


def test_coreml_metadata_summary_normalizes_ios_op_names():
    metadata = {
        "generatedClassName": "KokoroVocoder",
        "specificationVersion": 8,
        "storagePrecision": "Mixed (Float16, Palettized (8 bits))",
        "computePrecision": "Mixed (Float16, Float32, Int32)",
        "availability": {"iOS": "17.0"},
        "userDefinedMetadata": {"com.github.apple.coremltools.version": "9.0"},
        "mlProgramOperationTypeHistogram": {
            "Ios17.conv": 54,
            "Ios17.instanceNorm": 42,
            "Ios16.constexprLutToDense": 101,
            "Tile": 2,
        },
        "inputSchema": [
            {
                "name": "asr",
                "dataType": "Float16",
                "shape": "[1, 512, 120]",
                "hasShapeFlexibility": "1",
                "shapeFlexibility": "1 x 512 x 1...2000",
            }
        ],
        "outputSchema": [
            {
                "name": "x_pre",
                "dataType": "Float16",
                "shape": "[]",
                "hasShapeFlexibility": "0",
            }
        ],
    }

    summary = summarize_coreml_metadata("laishere", Path("KokoroVocoder.mlpackage"), metadata)
    assert summary["focusOps"] == {
        "conv": 54,
        "instanceNorm": 42,
        "tile": 2,
        "constexprLutToDense": 101,
    }
    assert summary["inputSchema"][0]["shapeFlexibility"] == "1 x 512 x 1...2000"

    markdown = render_coreml_metadata_markdown([summary])
    assert "laishere" in markdown
    assert "asr:Float16 [1, 512, 120]" in markdown
    assert "constexprLutToDense=101" in markdown


def test_generator_cos_snake_trim_or_pad_last_dim():
    import numpy as np

    arr = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    trimmed = _trim_or_pad_last_dim(arr, 2)
    padded = _trim_or_pad_last_dim(arr, 5)

    assert trimmed.shape == (1, 2, 2)
    assert trimmed.tolist() == [[[0.0, 1.0], [3.0, 4.0]]]
    assert padded.shape == (1, 2, 5)
    assert padded[..., :3].tolist() == arr.tolist()
    assert np.count_nonzero(padded[..., 3:]) == 0


def test_waveform_alignment_detects_lag_and_affine_fix():
    import numpy as np

    x = np.linspace(0.0, 8.0 * np.pi, 2048, dtype=np.float32)
    reference = np.sin(x).astype(np.float32)
    candidate = np.concatenate(
        [np.zeros(23, dtype=np.float32), (reference[:-23] * 0.5 + 0.05).astype(np.float32)]
    )

    result = analyze_pair(reference, candidate, max_lag=64, downsample=2)

    assert result["lagged"]["lag_samples"] == 23
    assert result["raw"]["snr_db"] < 10.0
    assert result["lagged_affine"]["metrics"]["snr_db"] > 60.0
    assert result["lagged_affine"]["gain"] > 1.9


def test_completion_verifier_allows_mlx_3s_error_but_requires_iphone():
    records = {}
    for machine in ("m2-studio", "m2-air", "irvine-m1"):
        for impl in ("config-f-reference", "mlx-audio", "soniqo-speech-swift-kokoro"):
            for key in RUNTIME_BUCKETS:
                if impl == "mlx-audio" and key == "3s":
                    records[(machine, impl, key)] = {"status": "error"}
                else:
                    records[(machine, impl, key)] = {
                        "status": "ok",
                        "warm_wall_times_s": [0.1] * 5,
                        "cold_wall_time_s": 0.2,
                    }

    assert _check_mac_primary(records) == []
    iphone_errors = _check_iphone(records)
    assert iphone_errors == [
        f"missing signed iPhone result: iphone-12-pro/soniqo-speech-swift-kokoro-ios/{key}"
        for key in RUNTIME_BUCKETS
    ]


def test_completion_verifier_loads_result_payloads_and_preflight(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    record = result_record(
        impl="config-f-reference",
        framework="Swift + Core ML",
        hardware_target="ANE/Core ML",
        version="test",
        machine_id="m2-studio",
        input_key="3s",
        text="sample",
        voice="af_heart",
        cold_wall_time_s=0.2,
        warm_wall_times_s=[0.1] * 5,
        canonical_audio_duration_s=1.0,
        observed_audio_duration_s=1.0,
        output_sha256="abc",
        provenance={},
    )
    payload = result_file_payload(
        impl="config-f-reference",
        machine_id="m2-studio",
        records=[record],
        provenance={},
    )
    (results_dir / "results_config_f_reference_m2-studio.json").write_text(
        json.dumps(payload)
    )
    (results_dir / "ios_runner_preflight_latest.json").write_text(
        json.dumps({"ok": False, "blockers": ["DEVELOPMENT_TEAM is unset"]})
    )

    records, errors = _load_records(results_dir)
    preflight, preflight_errors = _check_preflight(results_dir)

    assert errors == []
    assert ("m2-studio", "config-f-reference", "3s") in records
    assert preflight and not preflight["ok"]
    assert preflight_errors == ["iOS preflight not ready: DEVELOPMENT_TEAM is unset"]
