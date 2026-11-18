"""Bounding-box similarities used in the FGS-YOLO experiments."""

import torch


def _geometry(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    iou = inter / (w1 * h1 + w2 * h2 - inter + eps)
    dw1 = (b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1)).abs()
    dw2 = (b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1)).abs()
    dh1 = (b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1)).abs()
    dh2 = (b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1)).abs()
    penalty = ((dw1 + dw2) / w2.abs() + (dh1 + dh2) / h2.abs()) / 4
    return iou, penalty


def focaler_iou(box1, box2, d=0.0, u=0.95):
    """Focaler-IoU similarity."""
    iou, _ = _geometry(box1, box2)
    return ((iou - d) / (u - d)).clamp(0, 1)


def piou2(box1, box2, lambda_=1.3, d=None, u=None):
    """PIoU v2 similarity; optionally apply Focaler interval remapping first."""
    iou, penalty = _geometry(box1, box2)
    if d is not None and u is not None:
        iou = ((iou - d) / (u - d)).clamp(0, 1)
    piou_v1_loss = 2 - iou - torch.exp(-(penalty**2))
    quality = torch.exp(-penalty)
    x = quality * lambda_
    return 1 - 3 * x * torch.exp(-(x**2)) * piou_v1_loss
