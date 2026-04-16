#!/usr/bin/env python3
"""Compare Python and Swift tensor dumps for audio parity debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402

STAGE_ORDER = [
    "tokens",
    "attention_mask",
    "ref_s",
    "pred_dur",
    "pred_dur_valid",
    "duration_d",
    "duration_t_en",
    "alignment",
    "en",
    "asr",
    "en_padded",
    "f0",
    "n",
    "f0_padded",
    "n_padded",
    "asr_padded",
    "x_pre",
    "x_pre_padded",
    "har",
    "har_padded",
    "waveform",
]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return 1.0 if np.allclose(a, b) else 0.0
    a64 = a.astype(np.float64).reshape(-1)
    b64 = b.astype(np.float64).reshape(-1)
    if not np.isfinite(a64).all() or not np.isfinite(b64).all():
        return None
    a_std = float(a64.std())
    b_std = float(b64.std())
    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.allclose(a64, b64) else 0.0
    return float(np.corrcoef(a64, b64)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a64 = a.astype(np.float64).reshape(-1)
    b64 = b.astype(np.float64).reshape(-1)
    if not np.isfinite(a64).all() or not np.isfinite(b64).all():
        return None
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom == 0.0:
        return 1.0 if np.allclose(a64, b64) else 0.0
    return float(np.dot(a64, b64) / denom)


def compare_tensor(
    name: str,
    reference: np.ndarray | None,
    candidate: np.ndarray | None,
    *,
    max_abs: float,
    min_corr: float,
) -> dict[str, Any]:
    if reference is None:
        return {"name": name, "status": "missing_reference"}
    if candidate is None:
        return {"name": name, "status": "missing_candidate", "reference_shape": list(reference.shape)}

    result: dict[str, Any] = {
        "name": name,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        result["status"] = "shape_mismatch"
        return result

    if np.issubdtype(reference.dtype, np.integer) and np.issubdtype(candidate.dtype, np.integer):
        equal = bool(np.array_equal(reference, candidate))
        result.update(
            {
                "status": "pass" if equal else "fail",
                "max_abs_error": int(np.max(np.abs(reference.astype(np.int64) - candidate.astype(np.int64))))
                if reference.size
                else 0,
            }
        )
        return result

    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    diff = ref - cand
    max_err = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_err = float(np.mean(np.abs(diff))) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    corr = _corr(ref, cand)
    cosine = _cosine(ref, cand)
    passes = finite and (max_err <= max_abs or (corr is not None and corr >= min_corr))
    result.update(
        {
            "status": "pass" if passes else "fail",
            "finite": finite,
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "rmse": rmse,
            "correlation": corr,
            "cosine_similarity": cosine,
        }
    )
    return result


def _ordered_names(reference_names: set[str], candidate_names: set[str], explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    names = list(STAGE_ORDER)
    extras = sorted((reference_names | candidate_names) - set(names))
    return [name for name in names if name in reference_names or name in candidate_names] + extras


def compare(args: argparse.Namespace) -> dict[str, Any]:
    ref_manifest, ref_tensors = load_tensor_dump(args.reference)
    cand_manifest, cand_tensors = load_tensor_dump(args.candidate)
    names = _ordered_names(
        set(ref_tensors),
        set(cand_tensors),
        [name.strip() for name in args.names.split(",") if name.strip()] if args.names else None,
    )
    results = [
        compare_tensor(
            name,
            ref_tensors.get(name),
            cand_tensors.get(name),
            max_abs=args.max_abs,
            min_corr=args.min_corr,
        )
        for name in names
    ]
    first_failure = next((item["name"] for item in results if item["status"] != "pass"), None)
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_metadata": ref_manifest.get("metadata", {}),
        "candidate_metadata": cand_manifest.get("metadata", {}),
        "thresholds": {"max_abs": args.max_abs, "min_corr": args.min_corr},
        "first_failing_boundary": first_failure,
        "results": results,
        "hnsf_ladder_command": "uv run python scripts/validate_hnsf_swift.py generate && uv run python scripts/validate_hnsf_swift.py compare",
    }
    if args.write_json:
        path = Path(args.write_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--names", default=None, help="Comma-separated tensor names. Defaults to stage order.")
    parser.add_argument("--max-abs", type=float, default=1e-3)
    parser.add_argument("--min-corr", type=float, default=0.999)
    parser.add_argument("--write-json", default=None)
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()

    report = compare(args)
    print("name,status,shape,max_abs_error,correlation,cosine_similarity")
    for item in report["results"]:
        shape = item.get("reference_shape") or item.get("candidate_shape") or []
        print(
            ",".join(
                [
                    item["name"],
                    item["status"],
                    "x".join(str(v) for v in shape),
                    _fmt_float(item.get("max_abs_error")),
                    _fmt_float(item.get("correlation")),
                    _fmt_float(item.get("cosine_similarity")),
                ]
            )
        )
    if report["first_failing_boundary"]:
        print(f"first_failing_boundary={report['first_failing_boundary']}")
    else:
        print("first_failing_boundary=none")

    if args.fail_on_difference and report["first_failing_boundary"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
