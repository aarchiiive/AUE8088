import json
from pathlib import Path

import numpy as np
import cv2

import torch

from detect import Annotator, colors
from models.yolo import Model
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, colorstr, check_dataset
from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.general import xywh2xyxy, scale_boxes, non_max_suppression

def visualize(
    img_lwir,
    img_rgb,
    preds,
    imgsz,
    names,
    hide_conf: bool = False,
    line_thickness: int = 2,
    save_path: str = None,
):
    """
    Run NMS on preds and draw boxes on paired LWIR/visible images.

    Args:
        img_lwir: torch.Tensor of shape [B, C, H, W] (LWIR image batch)
        img_rgb: torch.Tensor of shape [B, C, H, W] (RGB
        preds: torch.Tensor of shape [B, N, 6] (model output before NMS)
        imgsz: tuple (height, width) of the model input size
        names: list of class names
        conf_thres, iou_thres, max_det, classes, agnostic_nms: NMS params
        hide_conf: if True, hide confidence in the label
        line_thickness: bounding‑box line width

    Returns:
        A list of concatenated numpy images [H, 2*W, 3], one per batch item.
    """
    results = []
    # 2) For each image in the batch
    for i, det in enumerate(preds):
        if det is None or not len(det):
            results.append(None)
            continue

        # 4) Create Annotators
        annot_lwir = Annotator(img_lwir, line_width=line_thickness, example=str(names))
        annot_vis  = Annotator(img_rgb,  line_width=line_thickness, example=str(names))

        # 5) Rescale boxes from model input → original image
        img_shape = img_rgb.shape[:2]   # (H, W)
        img0_shape = imgsz       # (H0, W0), pad
        det[:, :4] = scale_boxes(img_shape, det[:, :4], img0_shape).round()

        # 6) Draw boxes & labels
        for *xyxy, conf, cls in reversed(det):
            if conf > 0.25:
                c = int(cls)
                label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                color = colors(c, True)
                annot_lwir.box_label(xyxy, label, color=color)
                annot_vis.box_label(xyxy,  label, color=color)

        # 7) Collect results and concatenate LWIR+VIS horizontally
        res_lwir = annot_lwir.result()
        res_vis  = annot_vis.result()
        combined = np.concatenate([res_lwir, res_vis], axis=1)
        results.append(combined)

        if save_path:
            combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, combined)

    return results

def group_by_image_name(preds):
    """
    Group predictions by the 'image_name' field.

    Args:
        preds (List[Dict[str, Any]]): List of prediction dictionaries.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping from image name to its predictions.
    """
    grouped = {}
    for p in preds:
        name = p['image_name']
        grouped.setdefault(name, []).append(p)
    return grouped

def xywh2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    x_min, y_min, width, height = bboxes.unbind(1)
    x1 = x_min
    y1 = y_min
    x2 = x_min + width
    y2 = y_min + height
    return torch.stack((x1, y1, x2, y2), dim=1)

if __name__ == "__main__":
    lwir_dir = Path('datasets/kaist-rgbt/test/images/lwir')
    rgb_dir = Path('datasets/kaist-rgbt/test/images/visible')

    pred_path = Path('/your/path/to/predictions.json')  # Path to your predictions JSON file
    save_dir = Path('results/best')

    save_dir.mkdir(parents=True, exist_ok=True)

    imgsz = (512, 640)  # (height, width)
    names = ['person', 'cyclist', 'people', 'person?']

    with open(pred_path, 'r') as f:
        preds = json.load(f)

    print(f"Loaded {len(preds)} predictions from {pred_path}")

    grouped = group_by_image_name(preds)

    for image_name, preds in grouped.items():
        lwir_path = lwir_dir / f"{image_name}.jpg"
        rgb_path = rgb_dir / f"{image_name}.jpg"

        if len(preds) == 0:
            print(f"No predictions for {image_name}, skipping...")
            continue

        img_lwir = cv2.imread(str(lwir_path))
        img_rgb = cv2.imread(str(rgb_path))
        img_lwir = cv2.cvtColor(img_lwir, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)    # Convert to RGB
        save_path = save_dir / f"{image_name}.jpg"

        bboxes_xywh = torch.tensor([p['bbox'] for p in preds], dtype=torch.float32)  # [n, 4]
        classes = torch.tensor([p['category_id'] for p in preds], dtype=torch.float32).unsqueeze(1)  # [n, 1]
        confs   = torch.tensor([p['score'] for p in preds], dtype=torch.float32).unsqueeze(1)       # [n, 1]

        bboxes_xyxy = xywh2xyxy(bboxes_xywh)  # [n, 4]

        det = [torch.cat([bboxes_xyxy, confs, classes], dim=1)] # [n, 6]

        _ = visualize(
            img_lwir,
            img_rgb,
            det,
            imgsz,
            names,
            hide_conf=False,
            line_thickness=2,
            save_path=save_path
        )