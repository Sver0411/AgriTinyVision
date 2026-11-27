# Independent offline MobileNetV2 backend

The backend is a 224×224 MobileNetV2 classifier fine-tuned by `linkanjarad` from `google/mobilenet_v2_1.0_224` on a Kaggle distribution of PlantVillage containing 38 crop-health classes. The author reports six epochs of Adam training with batch size 256 and a learning rate of 0.00005. The separately trained AgriTinyVision edge model uses the same architecture family, torchvision ImageNet initialization, a 160×160 input and a binary output. The project runs the backend locally with the ONNX Runtime CPU provider. Its measurements cover payload bytes and model predictions. Radio energy, uplink latency, server queuing and cloud costs were outside the experiment. [Backend author’s model card at the original revision](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification/blob/c1861579a670fb6232258805b801cd4137cb7176/README.md) · [Independent ONNX conversion card](https://huggingface.co/onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX/blob/95110a1ffcc188d1b5a36524d52e303e40c2403d/README.md)

| Source artifact | SHA-256 |
| --- | --- |
| [Full float32 model](https://huggingface.co/onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX/resolve/95110a1ffcc188d1b5a36524d52e303e40c2403d/onnx/model.onnx) | `c209f96f0a87a265b0ff6765c2e84d48038a7d8b8692d7629445b4513c344eb7` |
| [Class mapping](https://huggingface.co/onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX/resolve/95110a1ffcc188d1b5a36524d52e303e40c2403d/config.json) | `788c73ea910607b69dd2fa71e1fde2c7f01b27741c2caae4ae4e6ecb14c22fcc` |
| [Image processor](https://huggingface.co/onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX/resolve/95110a1ffcc188d1b5a36524d52e303e40c2403d/preprocessor_config.json) | `3cb844ced1cab9e229a6631051eeb169e6edc92937e5b0c9652331679d3e00c5` |

The class mapping supplies the binary score. `p_diseased` sums the softmax probabilities of every label other than a `Healthy ...` class. The common binary threshold is 0.5 inclusive. The optional `p_diseased_known_crop` renormalizes logits within the known species for the four target crops. It is a secondary input condition. The primary score operates on the image alone and reads none of the reference labels.

Both transmission settings convert EXIF-oriented images to RGB and downscale proportionally to a maximum long edge of 224 or 112 pixels, with Lanczos interpolation and no enlargement. Pillow encodes baseline JPEG at quality 75 and 4:2:0 chroma subsampling. The backend decodes those bytes, resizes the short edge to 256 with bilinear interpolation, takes a centered 224×224 crop, and normalizes each channel to [−1,1]. `jpeg_bytes` gives the exact encoded image size. Each SHA-256 identifies the stored JPEG. Repeated images can share a stored file. Each transmission of a repeated image still incurs its full row-level size in the analyses.

## Image sources

PlantDoc uses revision `5467f6012d78d1c446145d5f582da6096f852ae8`. Reads use the original Git blobs to preserve case-sensitive path and byte identities. The prediction table contains both payload sizes for all 236 official test images and 1,242 official training images from apple, bell pepper, potato and tomato. The primary four-crop official test includes 131 images. The training images are separate records. Disease status follows the source folder taxonomy. [Released PlantDoc tree](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8) · [PlantDoc paper](https://doi.org/10.1145/3371158.3371196)

Cornell FGVC7 uses all 1,821 publicly labeled original `Train_N` images with their source one-hot diagnoses. Healthy maps to 0, while multiple diseases, rust, and scab map to 1. Here the benchmark’s labeled competition training split serves as an external evaluation cohort for frozen classifiers. Its acquisition and cross-checked label digests are in `data/fgvc_source.json`. [FGVC7 organizers](https://sites.google.com/view/fgvc7/competitions/plantpathology) · [Competition source](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)

## Reproduction

The run used Python 3.12, `numpy==2.3.5`, `Pillow==13.0.0` and `onnxruntime==1.32.0`. From the project root run the following commands.

```sh
python scripts/server_backend.py
python scripts/server_backend.py --fgvc --images data/raw/fgvc7 \
  --csv results/fgvc_backend.csv --payloads data/fgvc_backend_jpegs
python scripts/audit_backend.py
```

`results/backend_predictions.csv` and `results/fgvc_backend.csv` contain one row per source path and maximum long edge. Join within dataset by `(source_path, long_edge)`. Under `data/backend_jpegs` and `data/fgvc_backend_jpegs`, the filename is `jpeg_sha256` followed by `.jpg`. `results/backend_audit.json` reports confusion counts, AUROC with half credit for score ties, and upload sizes. The audit re-hashes all output JPEGs and joins the 60 archived ESP32 observations to their corresponding backend images using both source path and full source-file SHA-256.
