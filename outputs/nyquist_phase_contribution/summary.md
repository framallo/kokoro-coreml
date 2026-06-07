# Nyquist Phase Contribution Summary

PyTorch-only sensitivity probe for the compact `har_source -> waveform`
path. The table compares natural versus padded HAR geometry and whether
splicing the dumped Swift Nyquist phase is sufficient.

Rows: `10`.
Strict waveform gate pass rows: `5`.

| Bucket | Geometry | Dumped HAR SNR | Recomputed SNR | + dumped Nyquist SNR | Swift-branch Nyquist SNR | Swift-atan2 Nyquist SNR | Affine Nyquist SNR | Negated Nyquist SNR | Zero-Nyquist SNR | Nyquist wrapped max | 2pi errors | Swift-branch 2pi errors | Swift-atan2 2pi errors | Report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3s` | natural | 16.75 | 16.58 | 16.75 | 15.78 | 16.75 | 16.36 | 15.79 | 16.16 | 0.0872 | 2331 | 0 | 0 | `outputs/nyquist_phase_contribution/report_3s.json` |
| `3s` | padded | 50.06 | 26.45 | 50.06 | 25.59 | 50.06 | 26.46 | 21.78 | 24.83 | 0.0872 | 2331 | 0 | 0 | `outputs/nyquist_phase_contribution/report_3s_padded.json` |
| `7s` | natural | 16.01 | 15.75 | 16.00 | 15.49 | 16.00 | 15.63 | 15.45 | 15.59 | 0.3021 | 5487 | 0 | 0 | `outputs/nyquist_phase_contribution/report_7s.json` |
| `7s` | padded | 50.79 | 25.50 | 49.14 | 25.84 | 49.14 | 27.69 | 22.93 | 26.36 | 0.3021 | 5487 | 0 | 0 | `outputs/nyquist_phase_contribution/report_7s_padded.json` |
| `10s` | natural | 15.56 | 15.61 | 15.56 | 15.14 | 15.56 | 15.37 | 14.99 | 15.34 | 0.2466 | 7636 | 0 | 0 | `outputs/nyquist_phase_contribution/report_10s.json` |
| `10s` | padded | 49.95 | 26.25 | 49.87 | 26.55 | 49.87 | 27.64 | 22.95 | 25.98 | 0.2466 | 7636 | 0 | 0 | `outputs/nyquist_phase_contribution/report_10s_padded.json` |
| `15s` | natural | 16.13 | 15.86 | 16.13 | 15.59 | 16.13 | 15.62 | 15.37 | 15.53 | 1.2626 | 11433 | 0 | 0 | `outputs/nyquist_phase_contribution/report_15s.json` |
| `15s` | padded | 50.04 | 25.30 | 49.21 | 25.80 | 49.21 | 27.04 | 22.41 | 25.69 | 1.2626 | 11433 | 0 | 0 | `outputs/nyquist_phase_contribution/report_15s_padded.json` |
| `30s` | natural | 15.46 | 15.36 | 15.46 | 15.03 | 15.46 | 15.07 | 14.63 | 14.91 | 1.1840 | 23077 | 0 | 0 | `outputs/nyquist_phase_contribution/report_30s.json` |
| `30s` | padded | 48.63 | 24.79 | 48.42 | 25.41 | 48.42 | 26.36 | 21.59 | 24.89 | 1.1840 | 23077 | 0 | 0 | `outputs/nyquist_phase_contribution/report_30s_padded.json` |

## Interpretation

Using the raw trimmed waveform reference, dumped Nyquist phase plus padded shipping HAR geometry repairs the source-boundary path across 3s/7s/10s/15s/30s. Natural HAR geometry still fails strict waveform parity. Branch-only Swift-basis Nyquist repair fails, but exact Swift Float real/imag dot products followed by atan2 matches the dumped-Nyquist oracle. Prior fused-source timing still shows padded geometry removes the direct speed edge, so exact Nyquist repair is a strict contract unlock rather than a standalone production win.
