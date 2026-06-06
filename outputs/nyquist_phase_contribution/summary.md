# Nyquist Phase Contribution Summary

PyTorch-only sensitivity probe for the compact `har_source -> waveform`
path. The table compares natural versus padded HAR geometry and whether
splicing the dumped Swift Nyquist phase is sufficient.

Rows: `10`.
Strict waveform gate pass rows: `5`.

| Bucket | Geometry | Dumped HAR SNR | Recomputed SNR | + dumped Nyquist SNR | + dumped Nyquist corr | Delta SNR | Zero-Nyquist SNR | Nyquist wrapped max | 2pi errors | Report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3s` | natural | 16.75 | 16.58 | 16.75 | 0.988457 | 0.17 | 16.16 | 0.0872 | 2331 | `outputs/nyquist_phase_contribution/report_3s.json` |
| `3s` | padded | 50.06 | 26.45 | 50.06 | 0.999995 | 23.61 | 24.83 | 0.0872 | 2331 | `outputs/nyquist_phase_contribution/report_3s_padded.json` |
| `7s` | natural | 16.01 | 15.75 | 16.00 | 0.985984 | 0.25 | 15.59 | 0.3021 | 5487 | `outputs/nyquist_phase_contribution/report_7s.json` |
| `7s` | padded | 50.79 | 25.50 | 49.14 | 0.999993 | 23.64 | 26.36 | 0.3021 | 5487 | `outputs/nyquist_phase_contribution/report_7s_padded.json` |
| `10s` | natural | 15.56 | 15.61 | 15.56 | 0.984443 | -0.06 | 15.34 | 0.2466 | 7636 | `outputs/nyquist_phase_contribution/report_10s.json` |
| `10s` | padded | 49.95 | 26.25 | 49.87 | 0.999994 | 23.62 | 25.98 | 0.2466 | 7636 | `outputs/nyquist_phase_contribution/report_10s_padded.json` |
| `15s` | natural | 16.13 | 15.86 | 16.13 | 0.986450 | 0.27 | 15.53 | 1.2626 | 11433 | `outputs/nyquist_phase_contribution/report_15s.json` |
| `15s` | padded | 50.04 | 25.30 | 49.21 | 0.999993 | 23.91 | 25.69 | 1.2626 | 11433 | `outputs/nyquist_phase_contribution/report_15s_padded.json` |
| `30s` | natural | 15.46 | 15.36 | 15.46 | 0.984139 | 0.10 | 14.91 | 1.1840 | 23077 | `outputs/nyquist_phase_contribution/report_30s.json` |
| `30s` | padded | 48.63 | 24.79 | 48.42 | 0.999992 | 23.63 | 24.89 | 1.1840 | 23077 | `outputs/nyquist_phase_contribution/report_30s_padded.json` |

## Interpretation

Using the raw trimmed waveform reference, dumped Nyquist phase plus padded shipping HAR geometry repairs the source-boundary path across 3s/7s/10s/15s/30s. Natural HAR geometry still fails strict waveform parity, and prior fused-source timing shows padded geometry removes the speed edge, so Nyquist splicing is evidence for the blocker rather than a production win.
