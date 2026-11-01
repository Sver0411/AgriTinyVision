# Dataset audit

- created_at: 2026-10-05T18:14:48.935982+00:00
- total_images: 2792
- labels: {'NORMAL': 872, 'SUSPECT': 1920}
- class_ratios: {'NORMAL': 0.3123209169054441, 'SUSPECT': 0.6876790830945558}
- splits: {'train': 1964, 'test': 417, 'val': 411}
- split_labels: {'train': {'NORMAL': 619, 'SUSPECT': 1345}, 'test': {'NORMAL': 130, 'SUSPECT': 287}, 'val': {'NORMAL': 123, 'SUSPECT': 288}}
- source_classes: ['Apple___Apple_scab', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___healthy']
- missing_split_source_labels: []
- warnings: []
- source_coverage_rule: Any participating source class absent from train/val/test fails a split audit; unsplit or missing metadata warns.
- dimensions: {'256x256': 2792}
- invalid_count: 0
- near_hash: 64-bit DCT pHash
- near_distance: 4
- passed: True
- limitation: pHash is a heuristic; leaf IDs do not guarantee independent plant/scene/capture-session IDs.

## Source classes by frozen split

| Source class | train | val | test |
| --- | ---: | ---: | ---: |
| Apple___Apple_scab | 187 | 36 | 17 |
| Apple___Cedar_apple_rust | 163 | 37 | 40 |
| Apple___healthy | 182 | 36 | 22 |
| Pepper,_bell___Bacterial_spot | 175 | 35 | 30 |
| Pepper,_bell___healthy | 160 | 40 | 40 |
| Potato___Early_blight | 153 | 47 | 40 |
| Potato___Late_blight | 161 | 30 | 49 |
| Potato___healthy | 120 | 4 | 28 |
| Tomato___Early_blight | 178 | 30 | 32 |
| Tomato___Late_blight | 171 | 34 | 35 |
| Tomato___Leaf_Mold | 157 | 39 | 44 |
| Tomato___healthy | 157 | 43 | 40 |

Detailed offending indices are in the JSON report.
