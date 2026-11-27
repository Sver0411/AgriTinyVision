# Latvian apple scab source audit

The Institute of Horticulture of Latvia (LatHort) photographed apple leaves in Latvian orchards during the 2020 growing season. Its team manually sorted the photos according to visible apple scab. The archive has 207 `Healthy` images and 983 `Scab` images. The negative class can include mechanical damage, insects and other visible conditions. The authors captured multiple views of some leaves, so a source row identifies a photograph. [Institution and dataset links](https://www.darzkopibasinstituts.lv/en/projects/application-of-deep-learning-and-datamining-for-the-study-of-plant-pathogen-interaction) · [Acquisition protocol](https://agronomy.emu.ee/wp-content/uploads/2021/04/Vol19_No2_Kodors.pdf)

The official `AppleScabLDs` dataset was released on April 1, 2021 by **RTA & LatHort projects** and is distributed under CC BY-NC-ND 4.0. The 3,456,469,861-byte version-1 ZIP has SHA-256 `a56f3b825c30d5290094ba67f94a351b861fe583547a150a0c732597c50da6f4`; the provider's MD5 is `3cf7a8b891d28ac56b34b52966959b09`. The files in this directory contain metadata and hashes. The collection was fixed for evaluation before reviewing model outputs on it. [Dataset and attribution](https://www.kaggle.com/datasets/projectlzp201910094/applescablds) · [Official metadata](https://api.kaggle.com/v1/datasets/view/projectlzp201910094/applescablds) · [License](https://creativecommons.org/licenses/by-nc-nd/4.0/)

`blind_index.csv` is the frozen source index used as the `index` input to feature extraction. It preserves ZIP order, path, binary label, original folder, byte count and CRC32. Its SHA-256 is `9b191c73b76b7f3d476c0f5aa5d5ce937061422e23fec8b73ef6d1c94d470481`. Both audit modes rebuild it, verify that hash, and record it under `blind_index_sha256` in `provenance.json`.

`manifest.csv` preserves every path in archive order, with reference label, binary label (0 for `Healthy`, 1 for `Scab`), original byte count, CRC32, original SHA-256, EXIF-oriented width and height, and 64-bit perceptual hash. Its exact SHA-256 is `cf989140652bd94006ae703aba94d19effcf8f395278530b3525f47e5603afb8`. Original camera time, make and model are present for all 1,190 photos. There are 14 observed capture dates from June 8 through August 6, and nine dates include both labels. Date and device can support acquisition-group analyses. The source has one pair of byte-identical scab photos, yielding 1,189 unique file contents. Both original paths remain in the manifest.

`provenance.json` reports class and camera counts, identical source files, archive checksums and the overlap results. The audit compared the Latvian photos with 2,578 PlantDoc, 2,792 audited PlantVillage, 1,821 Cornell and 384 Kashmir reference photos. There were no exact SHA-256 matches or perceptual-hash candidates within five differing bits. This check covers the named reference images; proprietary or unreleased upstream pretraining collections are outside its scope.

`reference_fingerprints.csv.gz` holds the attribution group, SHA-256 and perceptual hash for each of those 7,575 reference images. It contains only these three fields. It was assembled in original row order from `data/features/source_fingerprints.csv` (`collection`, `file_sha256`, `phash_hex`), `results/fgvc_edge.csv` (`file_sha256`, `phash`, assigned `cornell`) and `results/transfer_manifest.csv` (`source_sha256`, `phash`, assigned `kashmir`). The pinned gzip SHA-256 is `d6bd7f43d7d4b93983a01b57f95fb89d9f6cc188a9bcf339b090546919bb07aa`.

## Rebuild from the original ZIP

Run the following commands from the project root. The audit was run with Python 3.12, Pillow 13.0.0 and NumPy 2.3.5. Keep the downloaded archive outside the project. Supply a pre-existing ZIP directly to `--archive` to skip the download step. The program checks its size and both archive digests, reads and CRC-checks each JPEG, recomputes the source metadata and perceptual hash, verifies the fixed manifest fingerprint, and writes `blind_index.csv`, `manifest.csv` and `provenance.json` under `--out`.

```sh
mkdir -p ../latvia-cache
curl -L --fail --retry 2 \
  'https://api.kaggle.com/v1/datasets/download/projectlzp201910094/applescablds?datasetVersionNumber=1' \
  -o ../latvia-cache/AppleScabLDs-v1.zip
python scripts/latvia_source_audit.py --archive ../latvia-cache/AppleScabLDs-v1.zip \
  --out ../latvia-cache/raw-audit
```

## Recheck the metadata cache

The following command reads the released manifest and its reference fingerprints without downloading or opening any photographs. It verifies the manifest hash and reconstructs the summary. In this mode archive-level hashes come from the pinned release constants. Actual examination of the archive requires `--archive`.

```sh
python scripts/latvia_source_audit.py --cached --out ../latvia-cache/cached-audit
```

Without `--out`, both modes write the metadata outputs to `data/latvia_source/`. Pass `--manifest PATH` in cached mode or `--refs PATH` in either mode to select another location for the same verified file. To read an individual original photo, use its `source_path` with the standard library's `ZipFile.read` and retain the original archive as the source of record.
