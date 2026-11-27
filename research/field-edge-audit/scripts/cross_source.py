"""Check Cornell source hashes against every fingerprint released for the two other corpora."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'data/features/source_fingerprints.csv').open() as f:
        reference=[(int(r['phash_hex'],16),r['file_sha256'],r['collection'],r['path']) for r in csv.DictReader(f)]
    with (ROOT/'results/fgvc_edge.csv').open() as f:
        orchard=list(csv.DictReader(f))
    if (len(reference),len(orchard)) != (5370,1821):
        raise ValueError('incomplete fingerprint tables')
    with (ROOT/'results/orchard_overlap.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['orchard_path','collection','other_path','file_identical','phash_distance'])
        for r in orchard:
            ph=int(r['phash'],16)
            for other,sha,collection,path in reference:
                distance=(ph^other).bit_count()
                same=sha==r['file_sha256']
                if distance<=4 or same:
                    writer.writerow([r['source_path'],collection,path,int(same),distance])


if __name__=='__main__':
    main()
