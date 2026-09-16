from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        preds = probs.view(logits.shape[0], -1)
        targets_flat = targets.view(logits.shape[0], -1)
        intersection = (preds * targets_flat).sum(dim=1)
        union = preds.sum(dim=1) + targets_flat.sum(dim=1)
        dice = ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
        loss = self.bce_weight * bce + self.dice_weight * (1.0 - dice)
        return loss


def _binarize(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (tensor >= threshold).to(dtype=torch.float32)


def dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = _binarize(preds, threshold)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return float(((2 * intersection + 1e-6) / (union + 1e-6)).mean().item())


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = _binarize(preds, threshold)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return float(((intersection + 1e-6) / (union + 1e-6)).mean().item())


def precision_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = _binarize(preds, threshold)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    return float(((tp + 1e-6) / (tp + fp + 1e-6)).mean().item())


def recall_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = _binarize(preds, threshold)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    tp = (preds * targets).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    return float(((tp + 1e-6) / (tp + fn + 1e-6)).mean().item())


def evaluate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    return {
        "dice": dice_score(probs, targets, threshold),
        "iou": iou_score(probs, targets, threshold),
        "precision": precision_score(probs, targets, threshold),
        "recall": recall_score(probs, targets, threshold),
    }
