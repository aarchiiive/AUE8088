# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import albumentations as A

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels

class AlbumentationsPair:
    """
    RGB/LWIR paired augmentations:
      - RGB: blur, gray, CLAHE, brightness/contrast, gamma, compression
      - LWIR: light blur + Gaussian noise to simulate sensor noise
    Bounding boxes (normalized xywh) are unchanged.
    """
    def __init__(self, size=640):
        # enforce Albumentations version
        check_version(A.__version__, "1.0.3", hard=True)
        prefix = colorstr("albumentations_pair: ")

        # RGB-only transforms (no geometry changes)
        rgb_transforms = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ImageCompression(quality_lower=75, p=0.01),
        ]
        # rgb_transforms = [
        #     A.Blur(blur_limit=3, p=0.02),               # Apply a 3×3 blur with 2% probability
        #     A.MedianBlur(blur_limit=3, p=0.02),         # Apply a 3×3 median blur with 2% probability
        #     A.ToGray(p=0.02),                           # Convert to grayscale with 2% probability
        #     A.CLAHE(clip_limit=2.0, p=0.02),            # Apply CLAHE with 2% probability
        #     A.RandomBrightnessContrast(p=0.3),          # Randomly adjust brightness/contrast with 30% probability
        #     A.RandomGamma(p=0.3),                       # Randomly adjust gamma with 30% probability
        #     A.HueSaturationValue(p=0.2),                # Shift hue/saturation/value with 20% probability
        #     A.ImageCompression(quality_lower=50, p=0.05) # Simulate JPEG compression artifacts with 5% probability
        # ]
        self.rgb_transform = A.Compose(rgb_transforms)

        # LWIR-only transforms: light blur + Gaussian sensor noise
        lwir_transforms = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.05),
        ]
        # LWIR-only transforms: stronger blur and sensor noise
        # lwir_transforms = [
        #     A.Blur(blur_limit=7, p=0.05),           # Apply a 7×7 blur with 5% probability
        #     A.MedianBlur(blur_limit=5, p=0.05),     # Apply a 5×5 median blur with 5% probability
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),  # Add Gaussian noise (variance 10–50) with 10% probability
        # ]
        self.lwir_transform = A.Compose(lwir_transforms)

        LOGGER.info(f"{prefix}RGB-only: " +
                    ", ".join(str(t) for t in rgb_transforms if t.p))
        LOGGER.info(f"{prefix}LWIR-only: " +
                    ", ".join(str(t) for t in lwir_transforms if t.p))

    def __call__(self, pair, labels, p=1.0):
        """
        Args:
            pair (tuple): (lwir_bgr, rgb_bgr), each H×W×3 uint8
            labels (np.ndarray[N,5]): (cls, x_center, y_center, w, h) normalized [0,1]
            p (float): probability to apply the augmentation set

        Returns:
            (lwir_aug, rgb_aug), labels (unchanged)
        """
        # unpack in LWIR–RGB order
        lwir, rgb = pair

        if random.random() < p:
            # apply RGB-specific transforms to the second element
            out_rgb = self.rgb_transform(image=rgb)
            rgb = out_rgb["image"]
            # apply LWIR-specific transforms to the first element
            out_lwir = self.lwir_transform(image=lwir)
            lwir = out_lwir["image"]

        # return in the same LWIR–RGB order
        return (lwir, rgb), labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """
    Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Example: y = (x - mean) / std
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """Equalizes image histogram, with optional CLAHE, for BGR or RGB image with shape (n,m,3) and range 0-255."""
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates half of the smallest object labels in an image for data augmentation.

    Returns augmented image and labels.
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets

def random_perspective_pair(
    img_lwir,
    img_rgb,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    """
    Apply the same random perspective/affine transform to an LWIR–RGB pair.

    Args:
        img_lwir (np.ndarray): H×W×3 LWIR image
        img_rgb  (np.ndarray): H×W×3 RGB image
        targets   (np.ndarray[N,5]): array of (cls, x1, y1, x2, y2) in pixel coords
        segments  (list[np.ndarray]): list of N polygon segments for copy-paste
        degrees, translate, scale, shear, perspective: augmentation hyperparameters
        border    (tuple): extra border (y, x) to allow for translation

    Returns:
        warped_lwir (np.ndarray): transformed LWIR image
        warped_rgb  (np.ndarray): transformed RGB image
        new_targets (np.ndarray): filtered & warped targets, same format as input
    """
    # image dimensions with optional border
    height = img_lwir.shape[0] + border[0] * 2
    width  = img_lwir.shape[1] + border[1] * 2

    # 1. Center translation to origin
    C = np.eye(3)
    C[0, 2] = -img_lwir.shape[1] / 2
    C[1, 2] = -img_lwir.shape[0] / 2

    # 2. Random perspective component
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # 3. Rotation + scale
    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D((0, 0), angle, s)

    # 4. Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # 5. Final translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined transformation matrix
    M = T @ S @ R @ P @ C

    # Apply to both modalities
    if perspective:
        warped_lwir = cv2.warpPerspective(img_lwir, M, (width, height), borderValue=(114, 114, 114))
        warped_rgb  = cv2.warpPerspective(img_rgb,  M, (width, height), borderValue=(114, 114, 114))
    else:
        warped_lwir = cv2.warpAffine(img_lwir, M[:2], (width, height), borderValue=(114, 114, 114))
        warped_rgb  = cv2.warpAffine(img_rgb,  M[:2], (width, height), borderValue=(114, 114, 114))

    # Transform bounding boxes
    n = len(targets)
    if n:
        # 4 corners per box in homogeneous coords
        pts = np.ones((n * 4, 3))
        pts[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        pts = pts @ M.T
        if perspective:
            pts[:, :2] /= pts[:, 2:3]
        pts = pts[:, :2].reshape(n, 8)

        # Reconstruct x1,y1,x2,y2
        x = pts[:, [0, 2, 4, 6]]
        y = pts[:, [1, 3, 5, 7]]
        new_boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip to valid image area
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)

        # Filter small or collapsed boxes
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        area1 = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        valid = area1 / (area0 + 1e-16) > 0.1

        targets = targets[valid].copy()
        targets[:, 1:5] = new_boxes[valid]

    return warped_lwir, warped_rgb, targets

def copy_paste(im, labels, segments, p=0.5):
    """
    Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.

    Details at https://arxiv.org/abs/2012.07177.
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments

def copy_paste_pair(img_lwir, img_rgb, labels, segments, p=0.5):
    """
    Apply YOLOv5-style copy-paste to an LWIR–RGB pair with exactly the same objects and locations.

    Args:
        img_lwir (np.ndarray): H×W×3 LWIR image
        img_rgb  (np.ndarray): H×W×3 RGB image
        labels   (np.ndarray): N×5 array of (cls, x1, y1, x2, y2)
        segments (list[np.ndarray]): list of N segment polygons (M_i×2 arrays)
        p        (float): fraction of segments to attempt to paste (0.0–1.0)

    Returns:
        img_lwir_out, img_rgb_out, labels_out, segments_out
    """
    n = len(segments)
    if not (p and n):
        return img_lwir, img_rgb, labels, segments

    h, w = img_lwir.shape[:2]
    mask_canvas = np.zeros((h, w), np.uint8)

    to_paste = random.sample(range(n), k=round(p * n))
    labels_out   = labels.copy()
    segments_out = segments.copy()

    for j in to_paste:
        cls, x1, y1, x2, y2 = labels[j]
        # horizontally flipped box
        box = (w - x2, y1, w - x1, y2)
        if (bbox_ioa(box, labels[:, 1:5]) < 0.30).all():
            labels_out   = np.vstack([labels_out,   [cls, *box]])
            seg = segments[j]
            flipped_seg = np.hstack([w - seg[:, :1], seg[:, 1:2]])
            segments_out.append(flipped_seg)
            cv2.drawContours(mask_canvas, [flipped_seg.astype(np.int32)], -1, 1, cv2.FILLED)

    # flip both images
    flipped_lwir = cv2.flip(img_lwir, 1)
    flipped_rgb  = cv2.flip(img_rgb,  1)
    mask = mask_canvas.astype(bool)

    # composite on mask
    img_lwir_out = img_lwir.copy()
    img_lwir_out[mask] = flipped_lwir[mask]
    img_rgb_out  = img_rgb.copy()
    img_rgb_out[mask]  = flipped_rgb[mask]

    return img_lwir_out, img_rgb_out, labels_out, segments_out

def cutout(im, labels, p=0.5):
    """
    Applies cutout augmentation to an image with optional label adjustment, using random masks of varying sizes.

    Details at https://arxiv.org/abs/1708.04552.
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending images and labels.

    See https://arxiv.org/pdf/1710.09412.pdf for details.
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

def mixup_pair(pair1, labels1, pair2, labels2):
    """
    MixUp for an LWIR–RGB pair.

    Args:
        pair1: tuple (img_lwir1, img_rgb1), each H×W×3 uint8
        labels1: np.ndarray[N1×5] (cls, x_center, y_center, w, h)
        pair2: tuple (img_lwir2, img_rgb2)
        labels2: np.ndarray[N2×5]

    Returns:
        mixed_pair: (mixed_lwir, mixed_rgb)
        mixed_labels: np.ndarray[(N1+N2)×5]
    """
    # same beta distribution for both modalities
    r = np.random.beta(32.0, 32.0)

    img_lwir1, img_rgb1 = pair1
    img_lwir2, img_rgb2 = pair2

    # blend both channels with the same ratio
    mixed_lwir = (img_lwir1 * r + img_lwir2 * (1 - r)).astype(np.uint8)
    mixed_rgb  = (img_rgb1  * r + img_rgb2  * (1 - r)).astype(np.uint8)

    # just concatenate labels
    mixed_labels = np.concatenate((labels1, labels2), axis=0)

    return (mixed_lwir, mixed_rgb), mixed_labels

def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """
    Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold
    `ar_thr`, and area ratio threshold `area_thr`.

    box1(4,n) is before augmentation, box2(4,n) is after augmentation.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Initializes a LetterBox object for YOLOv5 image preprocessing with optional auto sizing and stride
        adjustment.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads input image `im` (HWC format) to specified dimensions, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
