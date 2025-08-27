#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np


def read_csv_matrix(path: Path) -> np.ndarray:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(',')])
    return np.array(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True, help='Path to fixture JSON produced by tools/export_fixture.py')
    ap.add_argument('--swift_dir', required=True, help='Path to outputs/local/phase2_YYYYMMDD_HHMMSS')
    ap.add_argument('--out', help='Optional explicit output json path; defaults to swift_dir/inputs_parity.json')
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    swift_dir = Path(args.swift_dir)

    # Load arrays from fixture
    asr_ref = np.array(fixture['asr'], dtype=np.float32).reshape(fixture['shapes']['asr'])
    f0_ref = np.array(fixture['f0_curve'], dtype=np.float32).reshape(fixture['shapes']['f0_curve'])
    n_ref  = np.array(fixture['n'], dtype=np.float32).reshape(fixture['shapes']['n'])
    s_ref  = np.array(fixture['s'], dtype=np.float32).reshape(fixture['shapes']['s'])

    # Load arrays from Swift CSVs
    asr_csv = swift_dir / 'asr.csv'
    f0_csv  = swift_dir / 'f0_curve.csv'
    n_csv   = swift_dir / 'n.csv'
    s_csv   = swift_dir / 's.csv'

    if not asr_csv.exists():
        raise SystemExit(f"missing {asr_csv}. Re-run with KOKORO_DUMP_INPUTS=1.")

    asr_swift = read_csv_matrix(asr_csv).astype(np.float32)
    f0_swift  = read_csv_matrix(f0_csv).astype(np.float32)
    n_swift   = read_csv_matrix(n_csv).astype(np.float32)
    s_swift   = read_csv_matrix(s_csv).astype(np.float32)

    # Reshape fixture for comparison with CSV shapes
    asr_ref_2d = asr_ref.reshape(512, -1)
    f0_ref_2d  = f0_ref.reshape(1, -1)
    n_ref_2d   = n_ref.reshape(1, -1)
    s_ref_2d   = s_ref.reshape(1, -1)

    def mae(a, b):
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
        return float(np.mean(np.abs(a - b)))

    metrics = {
        'asr_mae': mae(asr_ref_2d, asr_swift),
        'f0_mae': mae(f0_ref_2d, f0_swift),
        'n_mae':  mae(n_ref_2d,  n_swift),
        's_mae':  mae(s_ref_2d,  s_swift),
    }

    out_path = Path(args.out) if args.out else (swift_dir / 'inputs_parity.json')
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote: {out_path}")


if __name__ == '__main__':
    main()
