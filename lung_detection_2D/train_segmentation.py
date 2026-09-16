from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import SegmentationDataset
from segmentation import DiceBCELoss, UNet, evaluate_metrics


def load_guidance_mapping(path: str) -> Dict[str, List[List[int]]]:
    with open(path, "r") as f:
        data = json.load(f)
    mapping = {}
    for key, boxes in data.items():
        mapping[str(key)] = [list(map(int, box)) for box in boxes]
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation model for LIDC central slice")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints/segmentation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--with-guidance", action="store_true")
    parser.add_argument("--guidance-json", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_loader(
    csv_path: str,
    images_dir: str,
    masks_dir: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    image_size: int,
    with_guidance: bool,
    guidance_mapping: Optional[Dict[str, List[List[int]]]],
) -> DataLoader:
    ds = SegmentationDataset(
        csv_path,
        images_dir,
        masks_dir,
        image_size=image_size,
        augment=shuffle,
        with_guidance=with_guidance,
        guidance_mapping=guidance_mapping,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    device_type = device.type if isinstance(device, torch.device) else str(device)
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=scaler is not None):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * images.shape[0]
    return total_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            batch_metrics = evaluate_metrics(logits, masks)
            for key, value in batch_metrics.items():
                metrics_sum[key] += value * images.shape[0]
    num_samples = len(loader.dataset)
    metrics = {k: v / num_samples for k, v in metrics_sum.items()}
    metrics["loss"] = total_loss / num_samples
    return metrics


def save_checkpoint(
    save_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: bool,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "best_dice": best,
    }
    torch.save(state, save_dir / ("best.pt" if best else "last.pt"))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    guidance_mapping = None
    if args.with_guidance and args.guidance_json is not None:
        guidance_mapping = load_guidance_mapping(args.guidance_json)

    train_loader = build_loader(
        args.train_csv,
        args.images_dir,
        args.masks_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        with_guidance=args.with_guidance,
        guidance_mapping=guidance_mapping,
    )
    val_loader = build_loader(
        args.val_csv,
        args.images_dir,
        args.masks_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_size=args.image_size,
        with_guidance=args.with_guidance,
        guidance_mapping=guidance_mapping,
    )

    model = UNet(in_channels=2 if args.with_guidance else 1, out_channels=1).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)
    scaler = torch.amp.GradScaler(enabled=args.amp and args.device.startswith("cuda"))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "runs" / time.strftime("%Y%m%d-%H%M%S")))

    start_epoch = 0
    best_dice = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_dice = checkpoint.get("best_dice", best_dice)

    stale = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args.device, scaler if args.amp else None)
        val_metrics = evaluate(model, val_loader, loss_fn, args.device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("val/precision", val_metrics["precision"], epoch)
        writer.add_scalar("val/recall", val_metrics["recall"], epoch)

        print(
            f"epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f}"
        )

        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            stale = 0

        save_checkpoint(save_dir, model, optimizer, epoch, best=is_best, args=args)
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_dice": best_dice,
            }, save_dir / "best.pt")
        if stale >= args.patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break

    writer.close()


if __name__ == "__main__":
    main()
