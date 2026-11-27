"""Fetch the pinned public files used by run_plantdoc.py (git, Python standard library)."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
AGRI = "https://github.com/Sver0411/AgriTinyVision"
PLANTDOC = "https://github.com/pratikkayal/PlantDoc-Dataset"
VILLAGE = "https://github.com/spMohanty/PlantVillage-Dataset"
AGRI_REV = "16e7a2fdec43c02258f22e85b085ac269345e124"
DOC_REV = "5467f6012d78d1c446145d5f582da6096f852ae8"
VILLAGE_REV = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"


def checkout(name, remote, revision, paths=None):
    directory = RAW / name
    if not (directory / ".git").exists():
        subprocess.run(["git","clone","--filter=blob:none","--depth=1","--no-checkout",remote+".git",str(directory)],check=True)
    git = ["git","-C",str(directory)]
    present = subprocess.run(git+["cat-file","-e",revision+"^{commit}"],capture_output=True)
    if present.returncode != 0:
        subprocess.run(git+["fetch","--depth=1","origin",revision],check=True)
    if paths:
        subprocess.run(git+["sparse-checkout","set","--no-cone","--stdin"],
                       input="\n".join(paths)+"\n",text=True,check=True)
    subprocess.run(git+["checkout","--quiet","--detach",revision],check=True)


def release_file(entry):
    name, target = entry
    destination = RAW/target
    if destination.exists():
        return
    url = AGRI+"/releases/download/v0.1.0/"+name
    with NamedTemporaryFile(dir=RAW,delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with urlopen(Request(url,headers={"User-Agent":"Field-Edge-Audit/1.0"}),timeout=120) as response, temp.open("wb") as output:
            shutil.copyfileobj(response,output)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    checkout("agritinyvision",AGRI,AGRI_REV,["/agritinyvision/","/configs/","/data/manifests/","/reports/field_validation/"])
    checkout("plantdoc",PLANTDOC,DOC_REV)
    manifest = json.loads((RAW/"agritinyvision/data/manifests/plantvillage_split.json").read_text())
    checkout("plantvillage",VILLAGE,VILLAGE_REV,["/raw/color/"+r["path"] for r in manifest["samples"]])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(release_file,[("tinyvision.onnx","tinyvision.onnx"),("metadata.json","tinyvision-metadata.json")]))
    if hashlib.sha256((RAW/"tinyvision.onnx").read_bytes()).hexdigest()!=json.loads((RAW/"tinyvision-metadata.json").read_text())["artifact_sha256"]["tinyvision.onnx"]:
        raise ValueError("Model did not match the upstream release hash")


if __name__=="__main__":
    main()
