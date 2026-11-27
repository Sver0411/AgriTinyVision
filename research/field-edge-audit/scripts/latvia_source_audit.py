"""Rebuild source-only Latvia fingerprints from a ZIP, or verify the released metadata cache."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import ExifTags, Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'data' / 'latvia_source'
ARCHIVE_BYTES = 3_456_469_861
ARCHIVE_SHA = 'a56f3b825c30d5290094ba67f94a351b861fe583547a150a0c732597c50da6f4'
ARCHIVE_MD5 = '3cf7a8b891d28ac56b34b52966959b09'
MANIFEST_SHA = 'cf989140652bd94006ae703aba94d19effcf8f395278530b3525f47e5603afb8'
BLIND_INDEX_SHA = '9b191c73b76b7f3d476c0f5aa5d5ce937061422e23fec8b73ef6d1c94d470481'
REFERENCES_SHA = 'd6bd7f43d7d4b93983a01b57f95fb89d9f6cc188a9bcf339b090546919bb07aa'
EXIF_FIELDS = ('exif_datetime_original', 'exif_make', 'exif_model')
SOURCE = 'https://www.kaggle.com/datasets/projectlzp201910094/applescablds'
DCT = np.cos((np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None] * (np.pi / 32)) * np.sqrt(2 / 32)
DCT[0] /= np.sqrt(2.)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive_check(path):
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('The source ZIP size differs from release 1')
    expected_sha, expected_md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as source:
        while chunk := source.read(4 * 1024 * 1024):
            expected_sha.update(chunk)
            expected_md5.update(chunk)
    if (expected_sha.hexdigest(), expected_md5.hexdigest()) != (ARCHIVE_SHA, ARCHIVE_MD5):
        raise ValueError('The source archive does not match the pinned SHA-256 and publisher MD5')


def exif_text(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value or '').strip('\x00 ')


def photo_record(archive, entry):
    source_path = entry.filename
    prefix, label, name = source_path.split('/')
    if prefix != 'AppleScabLDs' or label not in ('Healthy', 'Scab') or not name.lower().endswith('.jpg'):
        raise ValueError(f'Unexpected source member {source_path}')
    encoded = archive.read(entry)  # ZipFile verifies the original member's CRC while reading.
    with Image.open(BytesIO(encoded)) as source:
        exif = source.getexif()
        time = exif.get_ifd(ExifTags.IFD.Exif).get(36867, exif.get(36867, ''))
        camera = [exif_text(v) for v in (time, exif.get(271), exif.get(272))]
        image = ImageOps.exif_transpose(source).convert('RGB')
    gray = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float64)
    coefficients = (DCT @ gray @ DCT.T).flatten()
    phash = np.packbits(coefficients > np.median(coefficients[1:])).tobytes().hex()
    return dict(source_path=source_path, source_label=label, y=int(label == 'Scab'), source_sha256=sha(encoded),
                phash=phash, width=image.width, height=image.height, source_bytes=len(encoded), crc32=f'{entry.CRC:08x}',
                **dict(zip(EXIF_FIELDS, camera, strict=True)))


def manifest_from_zip(path):
    archive_check(path)
    text = StringIO(newline='')
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1190 or len(set(archive.namelist())) != 1190:
            raise ValueError('Expected 1,190 different archive member names')
        writer = None
        for entry in entries:
            row = photo_record(archive, entry)
            if writer is None:
                writer = csv.DictWriter(text, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
    return text.getvalue().encode('utf-8')


def compare_references(rows, source_file):
    compressed = source_file.read_bytes()
    if sha(compressed) != REFERENCES_SHA:
        raise ValueError('Reference fingerprint cache differs from the audited version')
    groups = defaultdict(list)
    for r in csv.DictReader(StringIO(gzip.decompress(compressed).decode('utf-8'))):
        groups[r['collection']].append((r['sha256'], int(r['phash'], 16)))
    comparisons = []
    for group, fingerprints in groups.items():
        digests = {digest for digest, _ in fingerprints}
        perceptual = [fingerprint for _, fingerprint in fingerprints]
        matches, least_distance, candidates = 0, 64, []
        for r in rows:
            matches += r['source_sha256'] in digests
            query = int(r['phash'], 16)
            distance = min((query ^ h).bit_count() for h in perceptual)
            least_distance = min(least_distance, distance)
            if distance <= 5:
                candidates.append(r['source_path'])
        comparisons.append(dict(reference=group, reference_n=len(fingerprints), exact_overlaps=matches,
                                pHash5_candidate_images=len(candidates), min_hamming=least_distance, paths=sorted(candidates)))
    return comparisons


def audit(manifest, reference):
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError('The per-image manifest differs from the independently fixed source audit')
    rows = list(csv.DictReader(StringIO(manifest.decode('utf-8'))))
    labels = Counter(r['source_label'] for r in rows)
    if labels != {'Healthy': 207, 'Scab': 983}:
        raise ValueError('Unexpected source labels')
    duplicates, dates = defaultdict(list), defaultdict(Counter)
    for r in rows:
        duplicates[r['source_sha256']].append(r['source_path'])
        dates[r['exif_datetime_original'][:10]][r['source_label']] += 1
    return dict(source=SOURCE, version=1, archive_bytes=ARCHIVE_BYTES, archive_sha256=ARCHIVE_SHA, labels=dict(labels),
                unique_content_sha256=len(duplicates), identical_content_groups=[v for v in duplicates.values() if len(v)>1],
                manifest_sha256=sha(manifest), license='CC BY-NC-ND 4.0', reference_overlap=compare_references(rows,reference),
                unit='Photo. Paper reports that individual leaves could be photographed from several viewpoints.',
                negative_label='Healthy folder signifies absence of visible scab; other biological/mechanical damage may be present.',
                published_archive_md5=ARCHIVE_MD5, exif_present={key: sum(bool(r[key]) for r in rows) for key in EXIF_FIELDS},
                exif_models=dict(Counter(r['exif_model'] for r in rows)), exif_date_label_counts={k:dict(v) for k,v in sorted(dates.items())},
                n_dates_with_both_labels=sum(bool(v['Scab']) and bool(v['Healthy']) for k,v in dates.items() if k))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--archive', type=Path, help='Official release-1 source ZIP, kept outside the package')
    mode.add_argument('--cached', action='store_true', help='Recheck the saved manifest and overlap counts without source images')
    p.add_argument('--refs', type=Path, default=ASSETS/'reference_fingerprints.csv.gz')
    p.add_argument('--manifest', type=Path, default=ASSETS/'manifest.csv', help='Manifest to read in --cached mode')
    p.add_argument('--out', type=Path, default=ASSETS)
    args = p.parse_args()
    manifest = args.manifest.read_bytes() if args.cached else manifest_from_zip(args.archive)
    report = audit(manifest, args.refs)
    index = StringIO(newline='')
    fields = ('source_path', 'y', 'source_label', 'source_bytes', 'crc32')
    writer = csv.DictWriter(index, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(csv.DictReader(StringIO(manifest.decode('utf-8'))))
    index_bytes = index.getvalue().encode('utf-8')
    if sha(index_bytes) != BLIND_INDEX_SHA:
        raise ValueError('The regenerated blind index differs from the frozen archive index')
    report['blind_index_sha256'] = BLIND_INDEX_SHA
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'manifest.csv').write_bytes(manifest)
    (args.out/'blind_index.csv').write_bytes(index_bytes)
    (args.out/'provenance.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Verified {sum(report['labels'].values())} rows and {sum(v['reference_n'] for v in report['reference_overlap'])} reference fingerprints")


if __name__ == '__main__':
    main()
