"""Package the local manuscript, predictions and source without third-party photographs."""

import hashlib
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'field-edge-audit-artifact.zip'
DIRECTORIES = ('scripts', 'tests', 'c_reference', 'paper', 'protocol', 'results',
               'data/features', 'data/latvia_source', 'data/raw/mahimahi_traces')
FILES = ('README.md', 'PUBLIC_DATA_INDEX.json', '.gitattributes', '.gitignore', 'field_pairs.py', 'requirements.txt', 'requirements-backend.txt')


def inputs():
    paths = set(ROOT / item for item in FILES)
    paths.update(p for p in (ROOT / 'data').iterdir() if p.is_file())
    for directory in DIRECTORIES:
        paths.update((ROOT / directory).rglob('*'))
    for path in sorted(paths):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            relative = path.relative_to(ROOT)
            if path.name in ('.DS_Store', 'latvia.npz') or path.suffix.lower() in ('.log', '.pyc'):
                continue
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.onnx', '.safetensors'):
                raise ValueError(f'Review before distributing this asset: {relative}')
            yield path


def main():
    temporary = ARCHIVE.with_suffix('.tmp.zip')
    hashes = []
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in inputs():
                name = path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                archive.writestr(name, data)
                hashes.append(f'{hashlib.sha256(data).hexdigest()}  {name}\n')
            archive.writestr('SHA256SUMS', ''.join(hashes))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Archive failed integrity check')
        os.replace(temporary, ARCHIVE)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{ARCHIVE} ({len(hashes)} files, {ARCHIVE.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
