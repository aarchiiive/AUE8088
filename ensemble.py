import os
import json
import uuid
from datetime import datetime
import numpy as np
from glob import glob
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.ops import nms
from ensemble_boxes import weighted_boxes_fusion, soft_nms


W, H = 640, 512


def _clip01(vals):
    return [min(max(v, 0.0), 1.0) for v in vals]


def xywhc_to_xyxy(box):
    """center‑xywh → corner‑xyxy"""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def xyxy_to_xywhc(x1, y1, x2, y2):
    """corner‑xyxy → center‑xywh"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [cx, cy, x2 - x1, y2 - y1]


def load_predictions(json_paths, path_format=None):
    """
    Load multiple JSON prediction files.
    Automatically applies path_format if paths are incomplete.

    Args:
        json_paths (list of str): filenames or full paths
        path_format (str or None): e.g., "runs/train/{}/preds/{}"

    Returns:
        models: list of dicts { image_id: [det, ...], ... }
        image_id_to_name: { image_id: image_name }
    """
    models = []
    image_id_to_name = {}

    for path in json_paths:
        path = path.strip()
        if not path or path.startswith("#") or path.startswith("{"):
            continue

        # Ensure path is a Path object
        path_obj = Path(path)

        # If file doesn't exist and format is given, try to build full path
        if path_format and not path_obj.exists():
            folder = path_obj.stem.split("_epoch")[0]
            try:
                path_obj = Path(path_format.format(folder, path_obj.name))
            except Exception as e:
                print(f"[Warning] Could not format path: {path}\n  Error: {e}")
                continue

        if not path_obj.exists():
            print(f"[Error] File not found: {path_obj}")
            continue

        try:
            with path_obj.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load: {path_obj}\n  Error: {e}")
            continue

        per_image = defaultdict(list)
        for det in data:
            img_id = det["image_id"]
            per_image[img_id].append(det)
            image_id_to_name.setdefault(img_id, det["image_name"])
        models.append(per_image)

    return models, image_id_to_name


def ensemble_wbf(models, image_id_to_name, iou_thr=0.5, skip_box_thr=0.0, weights=None):
    out = []
    for img_id in sorted(image_id_to_name):
        boxes_list, scores_list, labels_list = [], [], []

        # 1-2) center→corner→정규화→클립
        for m in models:
            dets = m.get(img_id, [])
            normed = []
            for d in dets:
                x1, y1, x2, y2 = xywhc_to_xyxy(d["bbox"])
                normed.append(_clip01([x1/W, y1/H, x2/W, y2/H]))
            boxes_list.append(normed)
            scores_list.append([d["score"] for d in dets])
            labels_list.append([d["category_id"] for d in dets])

        if not any(boxes_list):
            continue

        # 3) WBF
        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        # 4-5) 역정규화→corner→center
        for (nx1, ny1, nx2, ny2), s, l in zip(fb, fs, fl):
            x1, y1, x2, y2 = nx1*W, ny1*H, nx2*W, ny2*H
            out.append({
                "image_name":  image_id_to_name[img_id],
                "image_id":    img_id,
                "category_id": int(l),
                "bbox":        xyxy_to_xywhc(x1, y1, x2, y2),
                "score":       float(s)
            })
    return out


def ensemble_nms(models, image_id_to_name, iou_thr=0.5):
    out = []
    for img_id in sorted(image_id_to_name):
        boxes, scores, labels = [], [], []
        for m in models:
            for d in m.get(img_id, []):
                boxes.append(xywhc_to_xyxy(d["bbox"]))
                scores.append(d["score"])
                labels.append(d["category_id"])
        if not boxes:
            continue

        b = torch.tensor(boxes)
        s = torch.tensor(scores)
        keep = nms(b, s, iou_thr)
        for idx in keep.tolist():
            x1, y1, x2, y2 = boxes[idx]
            out.append({
                "image_name": image_id_to_name[img_id],
                "image_id":    img_id,
                "category_id": labels[idx],
                "bbox":        [ (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1 ],
                "score":       float(scores[idx])
            })
    return out


METHODS = {
    "wbf":    ensemble_wbf,
    "nms":    ensemble_nms,
}

if __name__ == "__main__":
    # Important: Make sure to install `ensemble_boxes` package
    # via `pip install ensemble-boxes` before running this script.

    # 1) find all input JSONs
    preds = [
        "/your/path/to/epoch1.json", # replace with your actual paths
        "/your/path/to/epoch2.json", # replace with your actual paths
        "/your/path/to/epoch3.json", #
    ]

    path_format = "runs/train/{}/preds/{}"
    models, id2name = load_predictions(preds, path_format)

    # 2) create a timestamped run folder under `ensembles/`
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250609_085346"
    run_dir = Path("ensembles") / f"{timestamp}_ensemble"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving ensembles to: {run_dir}/\n")

    # 3) for each method, make its subfolder and write results + sources
    for name, fn in METHODS.items():
        out_dir = run_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # run the ensemble
        result = fn(models, id2name)

        # write predictions JSON
        with open(out_dir / f"{timestamp}_ensemble_{name}.json", "w") as f:
            json.dump(result, f, indent=4)

        # write sources JSON
        with open(out_dir / f"{name}_sources.json", "w") as f:
            json.dump({"sources": [Path(p).name for p in preds]}, f, indent=4)

        print(f" • {name}:")
        print(f"    predictions → {out_dir / f'{name}.json'}")
        print(f"    sources     → {out_dir / f'{name}_sources.json'}")