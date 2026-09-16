from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import LIDCDetectionDataset, build_loader
from losses import YoloLoss, decode_predictions
from model import YoloLIDC
from utils import compute_map50, kmeans_anchors, nms, set_seed


# --------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _make_scheduler(optimizer, total_epochs: int, steps_per_epoch: int, warmup_epochs: int):
    warm_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warm_steps + 1, total_epochs * steps_per_epoch)
    warm = LambdaLR(optimizer, lr_lambda=lambda s: (s + 1) / warm_steps)
    cos = CosineAnnealingLR(optimizer, T_max=total_steps - warm_steps)
    return SequentialLR(optimizer, [warm, cos], milestones=[warm_steps])


# --------------------------------------------------------------------------
def _prepare_anchors(cfg: dict, train_ds: LIDCDetectionDataset) -> List[List[List[float]]]:
    """Return anchors as list-of-list-of-[w,h] (S=3, A=3)."""
    kmc = cfg.get("anchors_kmeans", {})
    if cfg["model"].get("anchors"):
        return cfg["model"]["anchors"]
    if not kmc.get("enabled", False):
        return cfg["model"]["anchors_fallback"]

    # collect wh from a subset of train samples
    n_samples = min(kmc.get("n_samples", 2000), len(train_ds))
    idxs = np.random.default_rng(0).choice(len(train_ds), n_samples, replace=False)
    wh = []
    for i in tqdm(idxs, desc="kmeans-anchors scan"):
        try:
            _, tgt = train_ds[int(i)]
        except Exception:
            continue
        for b in tgt["boxes"]:
            w = float(b[2] - b[0])
            h = float(b[3] - b[1])
            if w > 1 and h > 1:
                wh.append((w, h))
    if len(wh) < kmc.get("n_anchors", 9):
        print(f"[anchors] not enough boxes ({len(wh)}), falling back to defaults")
        return cfg["model"]["anchors_fallback"]
    wh_np = np.asarray(wh, dtype=np.float32)
    k = kmc.get("n_anchors", 9)
    centers = kmeans_anchors(wh_np, k=k, n_iter=kmc.get("n_iters", 30))
    groups = centers.reshape(3, -1, 2).tolist()
    print(f"[anchors] computed: {groups}")
    return groups


# --------------------------------------------------------------------------
def _post_process(
    preds: List[torch.Tensor],
    anchors: torch.Tensor,
    strides,
    conf_thr: float,
    iou_thr: float,
    max_det: int,
) -> List[Dict[str, torch.Tensor]]:
    dec = decode_predictions(preds, anchors, strides, conf_thr=conf_thr)
    outs = []
    for d in dec:
        if d["boxes"].shape[0] == 0:
            outs.append(d)
            continue
        keep = nms(d["boxes"], d["scores"], iou_thr=iou_thr)
        keep = keep[:max_det]
        outs.append({
            "boxes": d["boxes"][keep],
            "scores": d["scores"][keep],
            "mal_probs": d["mal_probs"][keep],
        })
    return outs


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg) -> Dict[str, float]:
    model.eval()
    anchors = loss_fn.anchors
    all_pred = []
    all_tgt = []
    mal_pred, mal_true = [], []

    infer = cfg["infer"]
    for imgs, targets in tqdm(loader, desc="eval"):
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        dets = _post_process(
            preds, anchors, loss_fn.strides,
            conf_thr=infer["conf_threshold"], iou_thr=infer["iou_threshold"],
            max_det=infer["max_det"],
        )
        for det, tgt in zip(dets, targets):
            all_pred.append({
                "boxes": det["boxes"].cpu().numpy(),
                "scores": det["scores"].cpu().numpy(),
            })
            all_tgt.append({
                "boxes": tgt["boxes"].cpu().numpy(),
            })
            # match each GT to best pred for malignancy metric
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_mal = tgt["malignancy"].cpu().numpy()
            p_boxes = det["boxes"].cpu().numpy()
            p_mal = det["mal_probs"].cpu().numpy()
            if len(gt_boxes) == 0 or len(p_boxes) == 0:
                continue
            from utils import _iou_np
            ious = _iou_np(gt_boxes, p_boxes)
            best = ious.argmax(axis=1)
            best_iou = ious.max(axis=1)
            for j, (bi, iou) in enumerate(zip(best, best_iou)):
                if iou >= 0.5:
                    mal_true.append(int(gt_mal[j]))
                    mal_pred.append(float(p_mal[bi]))

    map50 = compute_map50(all_pred, all_tgt)
    metrics = {"mAP50": float(map50)}
    if mal_pred:
        mp = np.asarray(mal_pred)
        mt = np.asarray(mal_true)
        acc = float(((mp >= 0.5).astype(int) == mt).mean())
        metrics["mal_accuracy"] = acc
        try:
            from sklearn.metrics import roc_auc_score, confusion_matrix
            if len(np.unique(mt)) == 2:
                metrics["mal_auc"] = float(roc_auc_score(mt, mp))
            cm = confusion_matrix(mt, (mp >= 0.5).astype(int), labels=[0, 1])
            metrics["mal_confusion"] = cm.tolist()
        except Exception:
            pass
    return metrics


# --------------------------------------------------------------------------
def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["train"]["seed"])

    ds_train = LIDCDetectionDataset(
        cfg["data"]["train_csv"], cfg["data"]["images_dir"], cfg["data"]["masks_dir"],
        image_size=cfg["data"]["image_size"], augment=True,
        col_patient=cfg["data"]["col_patient"], col_slice=cfg["data"]["col_center_slice"],
        col_malignancy=cfg["data"]["col_malignancy"], col_nodule_idxs=cfg["data"]["col_nodule_idxs"],
        aug_cfg=cfg["aug"],
    )
    ds_val = LIDCDetectionDataset(
        cfg["data"]["test_csv"], cfg["data"]["images_dir"], cfg["data"]["masks_dir"],
        image_size=cfg["data"]["image_size"], augment=False,
        col_patient=cfg["data"]["col_patient"], col_slice=cfg["data"]["col_center_slice"],
        col_malignancy=cfg["data"]["col_malignancy"], col_nodule_idxs=cfg["data"]["col_nodule_idxs"],
    )
    train_loader = build_loader(ds_train, args.batch_size, True, args.num_workers)
    val_loader = build_loader(ds_val, args.batch_size, False, args.num_workers)

    anchors = _prepare_anchors(cfg, ds_train)

    model = YoloLIDC(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)

    loss_fn = YoloLoss(
        anchors=anchors, strides=cfg["model"]["strides"],
        box_weight=cfg["loss"]["box_weight"], obj_weight=cfg["loss"]["obj_weight"],
        mal_weight=cfg["loss"]["mal_weight"],
        obj_pos_weight=cfg["loss"]["obj_pos_weight"], mal_pos_weight=cfg["loss"]["mal_pos_weight"],
        image_size=cfg["data"]["image_size"],
    ).to(device)

    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, args.epochs, max(1, len(train_loader)), cfg["train"]["warmup_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["train"]["log_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir))

    best_map = -1.0
    patience = cfg["train"]["early_stopping_patience"]
    stale = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                preds = model(imgs)
                loss, stats = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            writer.add_scalar("train/loss", stats["loss"], global_step)
            writer.add_scalar("train/box", stats["box"], global_step)
            writer.add_scalar("train/obj", stats["obj"], global_step)
            writer.add_scalar("train/mal", stats["mal"], global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1
            pbar.set_postfix({k: f"{v:.3f}" if isinstance(v, float) else v for k, v in stats.items()})

        metrics = evaluate(model, val_loader, loss_fn, device, cfg)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[epoch {epoch+1}] {metrics}")

        # save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "anchors": anchors,
            "cfg": cfg,
            "metrics": metrics,
        }, save_dir / "last.pt")

        cur = metrics.get("mAP50", 0.0)
        if cur > best_map:
            best_map = cur
            stale = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "anchors": anchors,
                "cfg": cfg,
                "metrics": metrics,
            }, save_dir / "best.pt")
            print(f"  -> new best mAP50={best_map:.4f}")
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stopping (no improvement in {patience} epochs)")
                break

    writer.close()


# --------------------------------------------------------------------------
def test(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    anchors = ckpt["anchors"]

    ds_test = LIDCDetectionDataset(
        cfg["data"]["test_csv"], cfg["data"]["images_dir"], cfg["data"]["masks_dir"],
        image_size=cfg["data"]["image_size"], augment=False,
        col_patient=cfg["data"]["col_patient"], col_slice=cfg["data"]["col_center_slice"],
        col_malignancy=cfg["data"]["col_malignancy"], col_nodule_idxs=cfg["data"]["col_nodule_idxs"],
    )
    loader = build_loader(ds_test, args.batch_size, False, args.num_workers)

    model = YoloLIDC(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    loss_fn = YoloLoss(
        anchors=anchors, strides=cfg["model"]["strides"],
        image_size=cfg["data"]["image_size"],
    ).to(device)

    metrics = evaluate(model, loader, loss_fn, device, cfg)
    out = Path(cfg["train"]["save_dir"]) / "test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"metrics saved to {out}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--weights", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # CLI overrides
    if args.epochs is None:
        args.epochs = cfg["train"]["epochs"]
    if args.batch_size is None:
        args.batch_size = cfg["train"]["batch_size"]
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.num_workers is None:
        args.num_workers = cfg["train"]["num_workers"]
    if not args.amp:
        args.amp = cfg["train"]["amp"]

    if args.mode == "train":
        train(cfg, args)
    else:
        test(cfg, args)


if __name__ == "__main__":
    main()
