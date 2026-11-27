"""Retrieve three pinned MobileNetV2 backend files and validate each hash."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE='https://huggingface.co/onnx-community/mobilenet_v2_1.0_224-plant-disease-identification-ONNX/resolve/95110a1ffcc188d1b5a36524d52e303e40c2403d/'
FILES={'onnx/model.onnx':'c209f96f0a87a265b0ff6765c2e84d48038a7d8b8692d7629445b4513c344eb7',
       'config.json':'788c73ea910607b69dd2fa71e1fde2c7f01b27741c2caae4ae4e6ecb14c22fcc',
       'preprocessor_config.json':'3cb844ced1cab9e229a6631051eeb169e6edc92937e5b0c9652331679d3e00c5'}
root=Path(__file__).resolve().parents[1]/'data/raw/backend-model'


def main():
    root.mkdir(parents=True,exist_ok=True)
    for name,digest in FILES.items():
        dst=root/Path(name).name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==digest:continue
        with urlopen(BASE+name,timeout=180) as r:data=r.read()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('asset hash mismatch: '+name)
        tmp=dst.with_suffix('.part')
        tmp.write_bytes(data);tmp.replace(dst)


if __name__=='__main__':
    main()
