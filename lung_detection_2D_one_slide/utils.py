
from __future__ import annotations

import math
import os
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch


# ----------------------------- Reproducibility ------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- Bbox utilities -------------------------------
def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1 = (box1[..., 2] - box1[..., 0]).clamp(0) * (box1[..., 3] - box1[..., 1]).clamp(0)
    a2 = (box2[..., 2] - box2[..., 0]).clamp(0) * (box2[..., 3] - box2[..., 1]).clamp(0)
    return inter / (a1 + a2 - inter + eps)


def ciou(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = bbox_iou(pred_xyxy, tgt_xyxy, eps)

    cw = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2]) - torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    ch = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3]) - torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    c2 = cw.pow(2) + ch.pow(2) + eps

    p_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    p_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    t_cx = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
    t_cy = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2
    rho2 = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    pw = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=eps)
    ph = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=eps)
    tw = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]).clamp(min=eps)
    th = (tgt_xyxy[..., 3] - tgt_xyxy[..., 1]).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + v * alpha)


# ------------------------------- NMS ---------------------------------------
def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.45) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = bbox_iou(
            boxes_xyxy[i].unsqueeze(0).expand(rest.numel(), -1),
            boxes_xyxy[rest],
        )
        order = rest[ious <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


# ----------------------------- Mask -> bbox --------------------------------
def bbox_from_mask(mask_2d: np.ndarray) -> Tuple[int, int, int, int] | None:
    m = (mask_2d > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def bboxes_from_mask_multi(mask_2d: np.ndarray, min_area: int = 4) -> List[Tuple[int, int, int, int]]:
    m = (mask_2d > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((x, y, x + w, y + h))
    return out


# ---------------------------- K-means anchors ------------------------------
def _wh_iou(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    inter = np.minimum(wh1[:, None, 0], wh2[None, :, 0]) * np.minimum(
        wh1[:, None, 1], wh2[None, :, 1]
    )
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-9
    return inter / union


def kmeans_anchors(wh: np.ndarray, k: int = 9, n_iter: int = 30, seed: int = 0) -> np.ndarray:
    assert wh.shape[0] >= k, f"Need at least {k} samples, got {wh.shape[0]}"
    rng = np.random.default_rng(seed)
    idx = rng.choice(wh.shape[0], k, replace=False)
    centers = wh[idx].copy()

    for _ in range(n_iter):
        d = 1 - _wh_iou(wh, centers)
        assign = d.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = wh[assign == j]
            if len(members) == 0:
                new_centers[j] = centers[j]
            else:
                new_centers[j] = members.mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    # sort by area, group into 3 scales of 3 anchors
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    return centers[order]


# ---------------------------- Visualisation --------------------------------
def draw_detections(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray | None = None,
    mal_probs: np.ndarray | None = None,
) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        mal = float(mal_probs[i]) if mal_probs is not None else 0.0
        color = (0, 0, 255) if mal >= 0.5 else (0, 255, 0)  # red malign / green benign
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        parts = []
        if scores is not None:
            parts.append(f"conf {float(scores[i]):.2f}")
        if mal_probs is not None:
            parts.append(f"mal {mal:.2f}")
        label = " | ".join(parts)
        if label:
            cv2.putText(
                img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA,
            )
    return img


# --------------------------- Simple mAP@0.5 --------------------------------
def compute_map50(
    preds: List[dict],
    targets: List[dict],
) -> float:
    
    all_scores = []
    all_tp = []
    total_gt = 0
    for pred, tgt in zip(preds, targets):
        gt_boxes = tgt["boxes"]
        total_gt += len(gt_boxes)
        if len(pred["boxes"]) == 0:
            continue
        scores = pred["scores"]
        order = np.argsort(-scores)
        boxes = pred["boxes"][order]
        scores = scores[order]

        matched = np.zeros(len(gt_boxes), dtype=bool)
        for b, s in zip(boxes, scores):
            all_scores.append(float(s))
            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue
            ious = _iou_np(b[None, :], gt_boxes)[0]
            j = int(ious.argmax())
            if ious[j] >= 0.5 and not matched[j]:
                matched[j] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if total_gt == 0 or len(all_scores) == 0:
        return 0.0

    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (total_gt + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    # 11-point interpolated AP
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precisions[recalls >= r].max() if (recalls >= r).any() else 0.0
        ap += p / 11
    return float(ap)


def _iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union