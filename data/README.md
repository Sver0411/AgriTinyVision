# Dataset provenance and storage

Images live only under ignored `data/raw/` or a user-supplied local root. Do not commit images, augmentations, caches or archives. `manifests/` and `metadata/` may be versioned.

Source: [PlantVillage original repository](https://github.com/spMohanty/PlantVillage-Dataset), raw/color only, immutable commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` in the first run. The downloader retains SHA256 and upstream leaf ID per image; downloads are replayable from the pinned revision. Set `revision` in configs/dataset.yaml to replay that source on a fresh machine.

First run: 2,792 images, 12 source labels, 4 crops (apple, bell pepper, potato, tomato), NORMAL 872 / SUSPECT 1,920. Maximum 240 per source label; potato healthy only has 152 available mapped images. The subset is not prevalence representative. The official leaf map contains source-label inconsistencies/missing mappings (e.g. Apple Black Rot alias and no Grape healthy groups); this run chooses other explicit classes and excludes ambiguous/unmapped images rather than inventing plant IDs.

`python scripts/download_plantvillage.py` downloads; `python -m agritinyvision prepare-data` audits before splitting, joins exact SHA256 / pHash <=4 duplicate components with real leaf groups, and uses GroupShuffleSplit with seed 41. The 70/15/15 ratios apply to groups, so image counts vary. Split uses no model predictions or test performance. Label mapping is centralized in configs/label_mapping.yaml.

Audit: invalid labels/paths, path traversal, zero byte, unreadable image, image size distribution, class/split ratios, SHA256 duplicates and full pairwise 64-bit DCT pHash risk. The pHash threshold is heuristic, not proof of perfect deduplication. Original leaf IDs do not identify entire plants/scenes/sessions, so residual leakage risk remains. Acquisition sequence/plant IDs and independent camera-domain data are required for field validation.

Local import: supply a JSON object with dataset/version and samples containing `path` (relative to your root), `source_label`, `label`, `group_id`. Map source labels using configs/label_mapping.yaml; use real plant/source/sequence IDs when available. `prepare-data --manifest ... --root ... --output ...` creates hashes and splits; point training config at it. Keep the unsplit source manifest separate and do not repeatedly regenerate test splits after consulting results.

Data licensing: no original images are redistributed by this code repository. The upstream repository/model card should be consulted for image terms and attribution; its loader currently includes an explicitly tentative license declaration. The repository MIT license covers our code only and makes no claim to license upstream imagery or ImageNet weights. Citation: Mohanty, Hughes, Salathé (2016), [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).

Clean backgrounds, controlled illumination, limited species, sampling bias and domain shift mean these results establish a pipeline, not reliable field disease detection. Future data must come from Gateway A cameras and include independent plant/site/time grouping, background/no-plant examples and difficult exposure/blur conditions.
