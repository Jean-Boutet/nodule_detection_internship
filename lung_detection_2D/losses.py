from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ciou, xywh_to_xyxy


# --------------------------------------------------------------------------
def _decode_pred(pred: torch.Tensor, anchors: torch.Tensor, stride: int) -> torch.Tensor:
    """Decode raw head output to (B,A,H,W,6) tensor of xywh, obj, mal in image space."""
    B, _, H, W = pred.shape
    A = anchors.shape[0]
    p = pred.view(B, A, 6, H, W).permute(0, 1, 3, 4, 2).contiguous()    # (B,A,H,W,6)
    # grid
    yv, xv = torch.meshgrid(
        torch.arange(H, device=pred.device), torch.arange(W, device=pred.device), indexing="ij"
    )
    grid = torch.stack([xv, yv], dim=-1).float()                        # (H,W,2)

    xy = (p[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride            # (B,A,H,W,2)
    wh = (p[..., 2:4].sigmoid() * 2.0) ** 2 * anchors.view(1, A, 1, 1, 2)
    decoded = torch.cat([xy, wh, p[..., 4:5], p[..., 5:6]], dim=-1)
    return decoded


class YoloLoss(nn.Module):
    def __init__(
        self,
        anchors: Sequence[Sequence[Tuple[float, float]]],
        strides: Sequence[int],
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        mal_weight: float = 1.0,
        obj_pos_weight: float = 1.0,
        mal_pos_weight: float = 1.0,
        anchor_wh_thr: float = 4.0,
        image_size: int = 256,
    ):
        super().__init__()
        assert len(anchors) == len(strides), "anchors and strides length mismatch"
        # store anchors as (S, A, 2) float tensor  (pixel WH for input resolution)
        anchors_t = torch.tensor(anchors, dtype=torch.float32)          # (S,A,2)
        self.register_buffer("anchors", anchors_t)
        self.strides = tuple(strides)
        self.num_scales = len(strides)
        self.num_anchors = anchors_t.shape[1]
        self.image_size = image_size
        self.anchor_wh_thr = anchor_wh_thr

        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.mal_weight = mal_weight

        self.bce_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([obj_pos_weight]),
            reduction="mean"
        )

        self.bce_mal = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([mal_pos_weight]), reduction="mean"
        )

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _match_targets(
        self,
        targets: List[dict],
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        out = []
        for s_idx, (stride, anchor_wh) in enumerate(zip(self.strides, self.anchors)):
            grid = self.image_size // stride
            b_list, a_list, gy_list, gx_list, tbox_list, tmal_list = [], [], [], [], [], []
            for b_i, t in enumerate(targets):
                boxes = t["boxes"].to(device)         # (N,4) xyxy pixels
                mal = t["malignancy"].to(device)      # (N,)
                if boxes.numel() == 0:
                    continue
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
                wh = boxes[:, 2:] - boxes[:, :2]
                gxy = cxcy / stride

                gx = gxy[:, 0]
                gy = gxy[:, 1]

                gxi = gx.long()
                gyi = gy.long()
                # anchor matching by wh ratio
                # ratios: (N, A)
                ratios = wh[:, None, :] / anchor_wh[None, :, :]
                r = torch.max(ratios, 1.0 / ratios).max(dim=-1).values
                mask = r < self.anchor_wh_thr             # (N, A)
                if mask.any():
                    idx_n, idx_a = torch.where(mask)  # (P,), (P,)
                    offsets = (
                        (0, 0),
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                    )   

                    for dx, dy in offsets:

                        gx_off = gxi[idx_n] + dx
                        gy_off = gyi[idx_n] + dy

                        valid = (
                            (gx_off >= 0)
                            & (gx_off < grid)
                            & (gy_off >= 0)
                            & (gy_off < grid)
                        )

                        if valid.any():

                            b_list.append(torch.full_like(idx_n[valid], b_i))
                            a_list.append(idx_a[valid])
                            gx_list.append(gx_off[valid])
                            gy_list.append(gy_off[valid])
                            tbox_list.append(boxes[idx_n[valid]])
                            tmal_list.append(mal[idx_n[valid]])
            if b_list:
                out.append((
                    torch.cat(b_list), torch.cat(a_list),
                    torch.cat(gy_list), torch.cat(gx_list),
                    torch.cat(tbox_list, dim=0), torch.cat(tmal_list),
                ))
            else:
                # empty
                empty = torch.zeros(0, dtype=torch.long, device=device)
                out.append((empty, empty, empty, empty,
                            torch.zeros(0, 4, device=device),
                            torch.zeros(0, device=device)))
        return out

    # ---------------------------------------------------------------------
    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[dict],
    ) -> Tuple[torch.Tensor, dict]:
        assert len(preds) == self.num_scales
        device = preds[0].device

        matched = self._match_targets(targets, device)
        total_box, total_obj, total_mal = 0.0, 0.0, 0.0
        n_pos = 0

        for s_idx, (pred, stride) in enumerate(zip(preds, self.strides)):
            anchors = self.anchors[s_idx].to(device)                 # (A,2)
            B, _, H, W = pred.shape
            A = self.num_anchors
            p = pred.view(B, A, 6, H, W).permute(0, 1, 3, 4, 2).contiguous()   # (B,A,H,W,6)

            b, a, gy, gx, tbox, tmal = matched[s_idx]
            obj_target = torch.zeros((B, A, H, W), device=device, dtype=pred.dtype)

            if b.numel() > 0:
                # gather positives
                p_pos = p[b, a, gy, gx]                              # (P,6)
                xy = (p_pos[:, 0:2].sigmoid() * 2.0 - 0.5 + torch.stack([gx, gy], dim=-1).float()) * stride
                wh = (p_pos[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[a]
                pred_xyxy = xywh_to_xyxy(torch.cat([xy, wh], dim=-1))
                c = ciou(pred_xyxy, tbox)                            # (P,)
                box_loss = (1.0 - c).mean()
                total_box = total_box + box_loss

                # obj target uses IoU as soft label (YOLOv5 style)
                obj_target[b, a, gy, gx] = c.detach().clamp(0.0, 1.0).to(obj_target.dtype)

                # malignancy loss on positives only
                mal_loss = F.binary_cross_entropy_with_logits(
                    p_pos[:, 5], tmal.to(p_pos.dtype), reduction="mean"
                )
                total_mal = total_mal + mal_loss
                n_pos += b.numel()

            # objectness on ALL locations
            obj_logits = p[..., 4]
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits,
                obj_target,
                reduction="mean"
            )
            total_obj = total_obj + obj_loss

        n_scales = float(self.num_scales)
        box_l = total_box / n_scales if isinstance(total_box, torch.Tensor) else torch.tensor(0.0, device=device)
        obj_l = total_obj / n_scales
        mal_l = total_mal / n_scales if isinstance(total_mal, torch.Tensor) else torch.tensor(0.0, device=device)

        loss = self.box_weight * box_l + self.obj_weight * obj_l + self.mal_weight * mal_l
        return loss, {
            "loss": float(loss.detach().cpu()),
            "box": float(box_l.detach().cpu()) if isinstance(box_l, torch.Tensor) else 0.0,
            "obj": float(obj_l.detach().cpu()),
            "mal": float(mal_l.detach().cpu()) if isinstance(mal_l, torch.Tensor) else 0.0,
            "n_pos": n_pos,
        }


# --------------------------------------------------------------------------
def decode_predictions(
    preds: List[torch.Tensor],
    anchors: torch.Tensor,           # (S,A,2) pixel WH
    strides: Sequence[int],
    conf_thr: float = 0.25,
    max_det_per_scale: int = 300,
) -> List[dict]:
    """
    Decode raw head outputs to per-image detections.
    Returns list (len B) of dicts with keys boxes(xyxy), scores, mal_probs.
    """
    device = preds[0].device
    B = preds[0].shape[0]
    all_boxes = [[] for _ in range(B)]
    all_scores = [[] for _ in range(B)]
    all_mal = [[] for _ in range(B)]

    for s_idx, (pred, stride) in enumerate(zip(preds, strides)):
        A = anchors.shape[1]
        _, _, H, W = pred.shape
        p = pred.view(B, A, 6, H, W).permute(0, 1, 3, 4, 2).contiguous()
        yv, xv = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        grid = torch.stack([xv, yv], dim=-1).float()
        xy = (p[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
        wh = (p[..., 2:4].sigmoid() * 2.0) ** 2 * anchors[s_idx].view(1, A, 1, 1, 2).to(device)
        obj = p[..., 4].sigmoid()
        mal = p[..., 5].sigmoid()

        boxes_xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)     # (B,A,H,W,4)
        for b in range(B):
            score_b = obj[b]                                # (A,H,W)
            keep = score_b > conf_thr
            if keep.sum() == 0:
                continue
            bx = boxes_xyxy[b][keep]
            sc = score_b[keep]
            ml = mal[b][keep]
            if bx.shape[0] > max_det_per_scale:
                top = sc.topk(max_det_per_scale)
                bx = bx[top.indices]
                sc = top.values
                ml = ml[top.indices]
            all_boxes[b].append(bx)
            all_scores[b].append(sc)
            all_mal[b].append(ml)

    outputs = []
    for b in range(B):
        if all_boxes[b]:
            outputs.append({
                "boxes": torch.cat(all_boxes[b], dim=0),
                "scores": torch.cat(all_scores[b], dim=0),
                "mal_probs": torch.cat(all_mal[b], dim=0),
            })
        else:
            outputs.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "mal_probs": torch.zeros((0,), device=device),
            })
    return outputs