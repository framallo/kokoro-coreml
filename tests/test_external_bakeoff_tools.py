import csv
import hashlib
import json
from pathlib import Path

from scripts.external_bakeoff.create_listening_review import _write_decisions_csv
from scripts.external_bakeoff.ingest_ios_runner_result import _ingest_records
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
from scripts.external_bakeoff.validate_listening_decisions import validate_rows
from scripts.external_bakeoff.verify_external_bakeoff_completion import (
    _check_iphone,
    _check_mac_primary,
    _check_preflight,
    _load_records,
)
from scripts.create_f0_source_listening_pack import _write_decisions_csv as _write_f0_decisions_csv
from scripts.summarize_f0_source_candidates import (
    collect_rows as collect_f0_candidate_rows,
    load_decisions as load_f0_candidate_decisions,
    render_markdown as render_f0_candidate_markdown,
    summarize_report as summarize_f0_candidate_report,
)
from scripts.validate_f0_source_listening_decisions import validate_rows as validate_f0_rows


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

    assert m2_air_3s["outcome"] == "config-f-loses"
    assert m2_air_3s["best_impl"] == "laishere-kokoro-coreml"
    assert round(m2_air_3s["gap_pct"], 1) == 4.2
    assert m2_air_7s["outcome"] == "no-full-duration-result"
    assert m2_air_7s["best_impl"] is None
    assert iphone_3s["outcome"] == "config-f-missing"
    assert iphone_3s["best_impl"] == "soniqo-speech-swift-kokoro-ios"
    assert summary["absolute_fastest_verified"] is False

    markdown = render_frontier_markdown(summary, rows, min_duration_ratio=0.95)
    assert "Config F needs 4.2%" in markdown
    assert "Excluded Short Outputs" in markdown
    assert "| m2-air | 7s | Soniqo | 70.0 ms | 0.741 |" in markdown


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
