import hashlib
import json

from scripts.download_field_data import CLASSES, REVISION, download


def test_frozen_source_selection_blob_integrity_and_independent_labels(tmp_path, monkeypatch):
    import scripts.download_field_data as module

    content = b"synthetic transport fixture; not an agricultural result"
    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    tree = {"truncated": False, "tree": [{"type": "blob", "path": "test/" + label + "/source.jpg", "sha": digest}
                                         for label in CLASSES]}

    def fetch(url):
        if "api.github.com" in url:
            return json.dumps(tree).encode()
        return b"CC-BY-4.0 fixture" if url.endswith("LICENSE.txt") else content

    monkeypatch.setattr(module, "fetch", fetch)
    manifest = download(tmp_path / "download", per_class=1)
    assert len(manifest["samples"]) == 15 and manifest["revision"] == REVISION
    normal = {r["source_label"] for r in manifest["samples"] if r["label"] == "NORMAL"}
    assert normal == {"Apple leaf", "Bell_pepper leaf", "Tomato leaf"}
    assert all(r["source_split"] == "test" and r["git_blob_sha1"] == digest for r in manifest["samples"])
    assert manifest["quality_labels"] == "not supplied"
    assert (tmp_path / "download/selection.json").exists()
