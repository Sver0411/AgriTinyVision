# Field-Edge Audit — BASA (Orchard Camera Paid Supervision, Streaming Deferral & Quantization Audit)

> **Budget-Amortized Source-Anchored Adaptation (BASA)** for bandwidth-limited orchard cameras, maintained inside the unified [AgriTinyVision](https://github.com/Sver0411/AgriTinyVision) repository (`research/field-edge-audit`).

## Key Artifacts

- **COMPAG (Elsevier) Manuscript & Source**: [paper/manuscript_compag.pdf](paper/manuscript_compag.pdf) · [paper/manuscript_compag.tex](paper/manuscript_compag.tex) · [paper/references.bib](paper/references.bib)
- **BASA Controller & Budget-Clamped Baselines**: [scripts/basa_controller.py](scripts/basa_controller.py) · [scripts/clamped_baselines.py](scripts/clamped_baselines.py) · [tests/test_basa_controller.py](tests/test_basa_controller.py)
- **Aggregated Results & Paired 95% CIs**: [results/basa/summary.csv](results/basa/summary.csv) · [results/basa/paired_ci.csv](results/basa/paired_ci.csv) · [results/basa/clamped_baselines.csv](results/basa/clamped_baselines.csv)
- **C99 Reference Implementations**: [c_reference/packed_head.c](c_reference/packed_head.c) · [c_reference/purchase_stream.c](c_reference/purchase_stream.c)
- **ESP32-S3 N16R8 On-Chip Benchmark**: [esp32s3_bench/](esp32s3_bench/README.md) · [results/basa/esp32s3_hardware_bench.json](results/basa/esp32s3_hardware_bench.json)

## Verified Headline Results (36 Primary Settings, 25% Uplink = 1,700 B/frame)

| Method | All 36 | Prepaid (18) | Earned / Zero-Credit (18) | Cap=32 (18) | Cap=128 (18) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native (0 B) | 65.38% | 65.38% | 65.38% | 65.38% | 65.38% |
| Pilot-Audit-16 | 68.62% | 68.65% | 68.58% | 68.60% | 68.63% |
| Frozen (0 labels, remote review) | 68.93% | 68.98% | 68.88% | 68.93% | 68.93% |
| Always-OvA (ICML 2022) | 71.78% | 72.86% | 70.70% | 72.41% | 71.15% |
| Always-EB-rank (EdgeBoost, 2025) | 71.84% | 72.86% | 70.81% | 72.50% | 71.17% |
| Always-CE (fixed-cap) | 72.04% | 73.16% | 70.91% | 72.62% | 71.45% |
| Clamped-OvA ($k \le k_{\mathrm{half}}$) | 72.61% | 74.06% | 71.16% | 72.52% | 72.70% |
| Clamped-EB-rank ($k \le k_{\mathrm{half}}$) | 72.72% | 74.09% | 71.36% | 72.62% | 72.82% |
| Clamped-CE ($k \le k_{\mathrm{half}}$) | 72.92% | 74.46% | 71.38% | 72.74% | 73.10% |
| **BASA (Ours)** | **73.50%** | **74.49%** | **72.52%** | **73.17%** | **73.84%** |

- **Uplink bandwidth sweep (all 36 settings)**: BASA leads at scarce **10%** (70.49% vs. 69.40% Clamped-CE, +1.09 pp) and primary **25%** (73.50% vs. 72.92% Clamped-CE, +0.58 pp; +1.14 pp under earned credit), while Clamped-CE edges slightly ahead at abundant **50%** (75.56% vs. 75.29%).
- **ASAR automatic anchoring**: OOF selection chooses unanchored fitting ($\lambda=1$) in 79.8% of V2 and 74.8% of V2-grid runs, while automatically switching to source anchoring ($\lambda \in \{5, 25, \infty\}$) in **86.8%** of MobileNetV4 runs.
- **On-chip ESP32-S3 N16R8 silicon benchmark** (160 MHz, ESP-IDF v5.4.4, 1,000 runs): **0/16 bit-exact mismatches** (`max_abs_err = 0.0`), **93 / 93 / 99 µs** `EQ01` INT8 head inference (0.018% of backbone latency), **41 / 45 / 45 µs** causal 64-frame streaming gate, **50 µs** head parse, **565 µs** activation quantization, **1,836 B** persistent state (`1,292 B` head + `544 B` gate), and **0 B** PSRAM delta.

## Quick Verification (Standard Library Only)

```bash
python3 scripts/verify_basa_results.py
python3 scripts/verify_purchase_results.py
python3 scripts/verify_revision.py
```

## Full Reproduction (Python 3.12)

```bash
python -m unittest discover -s tests
PYTHONPATH=scripts python scripts/basa_controller.py
PYTHONPATH=scripts python scripts/clamped_baselines.py
PYTHONPATH=scripts python scripts/plot_compag_figures.py
```

## Data Provenance & Licensing

Raw images and restricted labels remain at their upstream providers ([AgriTinyVision](https://github.com/Sver0411/AgriTinyVision), [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset), [Cornell FGVC7](https://doi.org/10.1002/aps3.11390), [Kashmir](https://doi.org/10.17632/9m2dcb5mmr.3), [AppleScabLDs](https://doi.org/10.15159/ar.21.045), [COMMECT 5G](https://doi.org/10.5281/zenodo.16919567)). `.gitignore` excludes `data/raw/*`, build directories, and cached JPEGs to prevent redistributing restricted assets.
