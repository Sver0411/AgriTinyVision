# Numerical and network sensitivity

## Released device scales

The only physical records are the original 60 ESP32-S3 PlantDoc predictions and the released fixture. Retrieve the two files from AgriTinyVision v0.1.0, then run the host approximation against the recorded image identities in this research project. The first script requires prepared source photographs and the original graph specified in the README. The second uses local FP32 and simulated activation caches. The graph checks the scale JSON and ONNX file fingerprints. It uses the S3 signed power-of-two configuration and round-half-up at active quantization boundaries, with floating-point operators between boundaries. Hardware intermediates are unavailable. The final embedded fixture is explicitly compared and disagrees with the approximation. This prevents claims of exact kernel parity.

```sh
gh release download v0.1.0 -R Sver0411/AgriTinyVision -p tinyvision.info -p tinyvision.json -D data/raw/esp-ppq
python scripts/esp_quantized_replay.py data/raw/esp-ppq
python scripts/esp_head_drift.py
```

The head drift experiment keeps split identities, JPEG charges, cutoff and remote service fixed. It swaps host features underneath a frozen head, then refits the same head on the simulated calibration activations. See [board comparison](../results/esp_grid_board.csv), [fixture](../results/esp_grid_anchor.json) and [three-site summary](../results/esp_head_drift_summary.csv). There is no new hardware timing, total RAM or power measurement. [ESP-DL quantization guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html)

## Delivery of a requested review

Run `python scripts/rural_link.py data/raw/commect_forestry.xlsx` with locally restored benchmark inputs. The script downloads the published table if needed and checks its checksum. Nine fixed-distance observations supply UL/DL goodput and RTT. A single server processes the retrospectively chosen frames in a deterministic, label-independent arrival order. Late replies leave the local decision in force. Four combinations of alert deadlines, inter-arrival times, cloud service times and connection reuse are tabulated. Measured goodput already includes the forest transport stack. The synthetic 10/20 kbit/s stress test charges a 52-byte allowance per segment attempt, up to five attempts, 1.2 s RTT and 60 s disconnected in every 120. It is a specified stress case, not an empirical NB-IoT or orchard trace. Human labeling effort and the ideal transfer time at setup are reported separately. The Section 3 streaming selector and the Section 6 link-delivery experiments are distinct.

The extracted [KPI rows](../results/rural_measured_kpis.csv) reproduce selected columns with unit/field renaming and omit the original location coordinates. Source: Telenor (Norway), *Radio and Network KPI measurements for 5G Private Network in Forest Environment*, 2025. These extracted measurements retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). [Dataset and license](https://zenodo.org/records/16919567) · [Deadline results](../results/rural_deadline_summary.csv) · [Setup and human-time assumptions](../results/rural_setup_cost.csv)
