import yaml
from tqdm import tqdm
from pathlib import Path

import numpy as np
import cv2

import torch

from detect import Annotator, colors
from models.yolo import Model
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
from utils.general import check_img_size, colorstr, check_dataset
from utils.general import scale_boxes, non_max_suppression

def visualize(
    preds,
    imgs,
    shapes,
    names,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    classes=None,
    agnostic_nms: bool = False,
    hide_conf: bool = False,
    line_thickness: int = 2,
    save_path: str = None,
):
    """
    Run NMS on preds and draw boxes on paired LWIR/visible images.

    Args:
        preds: torch.Tensor of shape [B, N, 6] (model output before NMS)
        imgs: tuple (lwir_batch, vis_batch), each a torch.Tensor [B, C, H, W]
        shapes: list of tuples [(img0_shape, ratio_pad), ...] for each batch
        names: list of class names
        conf_thres, iou_thres, max_det, classes, agnostic_nms: NMS params
        hide_conf: if True, hide confidence in the label
        line_thickness: bounding‑box line width

    Returns:
        A list of concatenated numpy images [H, 2*W, 3], one per batch item.
    """
    # 1) Apply NMS
    dets = non_max_suppression(
        preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
    )

    results = []
    # 2) For each image in the batch
    for i, det in enumerate(dets):
        if det is None or not len(det):
            results.append(None)
            continue

        # 3) Prepare images: (C, H, W) → (H, W, C) numpy
        img_lwir = imgs[0][i].cpu().numpy().transpose(1, 2, 0)
        img_vis  = imgs[1][i].cpu().numpy().transpose(1, 2, 0)
        img_lwir = np.ascontiguousarray(img_lwir)
        img_vis  = np.ascontiguousarray(img_vis)

        # 4) Create Annotators
        annot_lwir = Annotator(img_lwir, line_width=line_thickness, example=str(names))
        annot_vis  = Annotator(img_vis,  line_width=line_thickness, example=str(names))

        # 5) Rescale boxes from model input → original image
        img_shape = img_vis.shape[:2]   # (H, W)
        img0_shape, _ = shapes[i]       # (H0, W0), pad
        det[:, :4] = scale_boxes(img_shape, det[:, :4], img0_shape).round()

        # 6) Draw boxes & labels
        for *xyxy, conf, cls in reversed(det):
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


if __name__ == "__main__":
    data = 'data/kaist-rgbt.yaml'
    hyp = 'data/hyps/hyp.scratch-low.yaml'

    modalities = ('lwir', 'visible')
    weights = '/your/path/to/weights/best.pt'
    cfg = 'models/yolov5n_kaist-rgbt.yaml'

    save_dir = Path(weights).parent.parent / 'vis' / 'test'  # save results to 'runs/vis/'
    save_dir.mkdir(parents=True, exist_ok=True)  # make directory if not exists

    data_dict = check_dataset(data)
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    device = select_device('cuda:3')

    imgsz = 640
    batch_size = 1
    single_cls = True
    nc = data_dict["nc"] if not single_cls else 1  # number of classes
    seed = 0

    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    model.load_state_dict(torch.load(weights, map_location='cpu')['model'].state_dict()) # IMPORTANT: load state_dict from the checkpoint

    test_path = data_dict["test"]
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    dataloader, dataset = create_dataloader(
        test_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        hyp=hyp,
        augment=False,      # TODO: check if there is no bug when applying augmentation
        cache=None,
        rect=False,
        rank=-1,
        workers=1,
        image_weights=False,
        quad=False,
        prefix=colorstr("test: "),
        shuffle=False,      # No shuffle for debugging
        seed=seed,
        rgbt_input=True,
        modalities=modalities,
    )

    # inference
    model.eval()
    for batch_i, (imgs, targets, paths, shapes, _) in enumerate(tqdm(dataloader)):
        ims = [im.to(device, non_blocking=True).float() / 255 for im in imgs]    # For RGB-T input
        with torch.no_grad():
            pred = model(ims)  # forward -> [xyxy, conf, cls]
            _ = visualize(pred, imgs, shapes, data_dict["names"], save_path=str(save_dir / Path(paths[0]).name))

