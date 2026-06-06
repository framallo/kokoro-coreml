# Nyquist Phase Contribution Summary

PyTorch-only sensitivity probe for the compact `har_source -> waveform`
path. The table compares natural versus padded HAR geometry and whether
splicing the dumped Swift Nyquist phase is sufficient.

Rows: `10`.
Strict waveform gate pass rows: `2`.

| Bucket | Geometry | Dumped HAR SNR | Recomputed SNR | + dumped Nyquist SNR | + dumped Nyquist corr | Delta SNR | Zero-Nyquist SNR | Nyquist wrapped max | 2pi errors | Report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3s` | natural | 16.74 | 16.58 | 16.74 | 0.988453 | 0.17 | 16.16 | 0.0872 | 2331 | `outputs/nyquist_phase_contribution/report_3s.json` |
| `3s` | padded | 47.76 | 26.44 | 47.76 | 0.999991 | 21.32 | 24.82 | 0.0872 | 2331 | `outputs/nyquist_phase_contribution/report_3s_padded.json` |
| `7s` | natural | 16.01 | 15.75 | 16.00 | 0.985984 | 0.25 | 15.59 | 0.3021 | 5487 | `outputs/nyquist_phase_contribution/report_7s.json` |
| `7s` | padded | 50.78 | 25.50 | 49.13 | 0.999993 | 23.64 | 26.36 | 0.3021 | 5487 | `outputs/nyquist_phase_contribution/report_7s_padded.json` |
| `10s` | natural | 11.63 | 11.70 | 11.63 | 0.962793 | -0.07 | 11.58 | 0.2466 | 7636 | `outputs/nyquist_phase_contribution/report_10s.json` |
| `10s` | padded | 14.10 | 13.94 | 14.10 | 0.978626 | 0.16 | 13.92 | 0.2466 | 7636 | `outputs/nyquist_phase_contribution/report_10s_padded.json` |
| `15s` | natural | 12.59 | 12.50 | 12.59 | 0.970113 | 0.09 | 12.34 | 1.2626 | 11433 | `outputs/nyquist_phase_contribution/report_15s.json` |
| `15s` | padded | 15.27 | 14.92 | 15.27 | 0.983636 | 0.35 | 14.99 | 1.2626 | 11433 | `outputs/nyquist_phase_contribution/report_15s_padded.json` |
| `30s` | natural | 12.22 | 12.24 | 12.22 | 0.967396 | -0.02 | 12.02 | 1.1840 | 23077 | `outputs/nyquist_phase_contribution/report_30s.json` |
| `30s` | padded | 15.09 | 14.78 | 15.09 | 0.982891 | 0.31 | 14.80 | 1.1840 | 23077 | `outputs/nyquist_phase_contribution/report_30s_padded.json` |

## Interpretation

Dumped Nyquist phase repairs the 3s/7s padded source-boundary path, but it does not make natural geometry strict and the 10s/15s/30s direct Nyquist probe does not reproduce strict waveform parity even with dumped HAR. Treat long-bucket Nyquist results as evidence that the direct probe has a reference/geometry mismatch beyond Nyquist, not as a production win.
