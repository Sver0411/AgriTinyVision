"""Human labels remain independent of device predictions and imported source labels."""

import csv
import io
import json

from .capture import now, read_samples

FIELDS = ("sample_id", "plant_presence", "image_quality", "visual_label", "judgment_certain", "crop_type",
          "environment", "lighting", "distance", "angle", "background", "abnormality_severity",
          "plant_id", "session_id", "disease_label", "disease_evidence", "expert_confirmed", "annotator", "notes")
QUALITY = ("USABLE", "BLUR", "TOO_DARK", "OVEREXPOSED", "LOW_CONTRAST", "OCCLUDED")
ENUMS = {"plant_presence": ("YES", "NO", "UNCERTAIN"), "image_quality": QUALITY,
         "visual_label": ("NORMAL", "SUSPECT", "UNCERTAIN", "NOT_APPLICABLE")}


def labels(batch, kind="manual"):
    path = batch / "annotations" / (kind + ".jsonl")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["sample_id"]: row for row in (json.loads(line) for line in f if line.strip())}


def save_annotation(batch, sample_id, value):
    if sample_id not in {s["sample_id"] for s in read_samples(batch)}:
        raise ValueError("样本不存在")
    for key, options in ENUMS.items():
        if value.get(key) not in options:
            raise ValueError("请填写 " + key)
    for key in ("judgment_certain", "expert_confirmed"):
        if type(value.get(key)) is not bool:
            raise ValueError("请填写 " + key)
    if value.get("disease_label") and not value.get("disease_evidence", "").strip():
        raise ValueError("病害标签需要可靠依据；模型输出不能作为依据")
    if value["plant_presence"] == "NO" and value["visual_label"] != "NOT_APPLICABLE":
        raise ValueError("无目标叶片时 visual_label 应为 NOT_APPLICABLE")
    row = {k: value.get(k, "") for k in FIELDS if k != "sample_id"}
    if any(not isinstance(v, (str, bool)) or isinstance(v, str) and len(v) > 2000 for v in row.values()):
        raise ValueError("标注字段应为文本或布尔值，文本至多 2000 字符")
    row.update(sample_id=sample_id, saved_at=now(), label_origin="manual")
    with (batch / "annotations/manual.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def annotation_csv(batch, kind="manual"):
    saved = labels(batch, kind)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, FIELDS, extrasaction="ignore")
    writer.writeheader()
    for sample in read_samples(batch):
        writer.writerow({"sample_id": sample["sample_id"], **saved.get(sample["sample_id"], {})})
    return output.getvalue()


def import_annotations(batch, filename):
    with open(filename, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("judgment_certain", "expert_confirmed"):
            if row[key].lower() not in ("true", "false"):
                raise ValueError(key + " must be true or false")
            row[key] = row[key].lower() == "true"
        save_annotation(batch, row["sample_id"], row)
    return len(rows)
