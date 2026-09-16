from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "lung_detection_2D_one_slide"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import bboxes_from_mask_multi


def _to_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    s = str(x).strip()
    if not s or s in ("[]", "nan", "None"):
        return []
    try:
        val = ast.literal_eval(s)
        return list(val) if isinstance(val, (list, tuple)) else [val]
    except Exception:
        return [t.strip() for t in s.strip("[]").split(",") if t.strip()]


def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        if a.shape[-1] in (1, 3):
            if a.shape[-1] == 1:
                return a[..., 0].astype(np.float32)
            return a[..., 1].astype(np.float32)
        if a.shape[0] in (1, 3):
            if a.shape[0] == 1:
                return a[0].astype(np.float32)
            return a[1].astype(np.float32)
    raise ValueError(f"Unsupported image shape: {a.shape}")


def _context_channels(arr: np.ndarray, in_channels: int) -> np.ndarray:
    slice2d = _normalize_slice(arr)
    if slice2d.max() > 1.5:
        slice2d = slice2d / max(float(slice2d.max()), 1e-6)
    slice2d = np.clip(slice2d, 0.0, 1.0).astype(np.float32)

    if in_channels == 1:
        return slice2d[None, :, :]

    # Practical 3D-aware input: emulate volumetric context as stacked channels, not a real 3D conv layout.
    channels = max(1, int(in_channels))
    if channels == 3:
        return np.stack([slice2d, slice2d, slice2d], axis=0).astype(np.float32)
    out = np.repeat(slice2d[None, :, :], channels, axis=0)
    return out.astype(np.float32)


class LIDC3DContextDataset(Dataset):
    """Practical 3D-aware dataset: keep the real slice preprocessing but provide an n-channel context stack."""

    def __init__(
        self,
        csv_path: str | os.PathLike,
        images_dir: str | os.PathLike,
        masks_dir: str | os.PathLike,
        image_size: int = 256,
        depth: int = 3,
        col_patient: str = "patient_id",
        col_slice: str = "center_slice",
        col_malignancy: str = "malignancy",
        col_nodule_idxs: str = "grouped_nodule_idxs",
        col_image_path: str = "image_path",
        col_mask_path: str = "mask_path",
        in_channels: int = 3,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.depth = max(1, int(depth))
        self.col_patient = col_patient
        self.col_slice = col_slice
        self.col_malignancy = col_malignancy
        self.col_nodule_idxs = col_nodule_idxs
        self.col_image_path = col_image_path
        self.col_mask_path = col_mask_path
        self.in_channels = int(in_channels)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row: pd.Series, root: Path, col_name: str) -> Path:
        if col_name in row and isinstance(row[col_name], str):
            p = Path(row[col_name])
            if not p.is_absolute():
                p = root / p
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not resolve path for {col_name}: {row[col_name]}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row, self.images_dir, self.col_image_path)
        mask_path = self._resolve_path(row, self.masks_dir, self.col_mask_path)

        image = _context_channels(np.load(img_path), self.in_channels)
        mask = _normalize_slice(np.load(mask_path))

        if image.shape[-2:] != (self.image_size, self.image_size):
            image = np.stack(
                [cv2.resize(c, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR) for c in image],
                axis=0,
            )
        mask = cv2.resize((mask > 0).astype(np.uint8), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        mask2d = (mask > 0).astype(np.uint8)
        if mask2d.shape != (self.image_size, self.image_size):
            mask2d = cv2.resize(mask2d, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        bxs = bboxes_from_mask_multi(mask2d, min_area=4)
        bxs.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        boxes_arr = np.asarray(bxs, dtype=np.float32).reshape(-1, 4) if bxs else np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((len(boxes_arr),), dtype=np.int64)
        mal_list = [int(v) for v in _to_list(row[self.col_malignancy])] if self.col_malignancy in self.df.columns else []
        if mal_list and len(mal_list) != len(boxes_arr):
            mal_list = (mal_list + [0] * len(boxes_arr))[: len(boxes_arr)]
        elif not mal_list:
            mal_list = [0] * len(boxes_arr)
        malignancy = np.asarray(mal_list, dtype=np.float32).reshape(-1)

        image_t = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32)))
        targets = {
            "boxes": torch.from_numpy(boxes_arr) if len(boxes_arr) else torch.zeros((0, 4)),
            "labels": torch.from_numpy(labels) if len(labels) else torch.zeros((0,), dtype=torch.long),
            "malignancy": torch.from_numpy(malignancy) if len(malignancy) else torch.zeros((0,)),
            "image_id": f"{row[self.col_patient]}_{int(row[self.col_slice])}",
            "orig_size": (self.image_size, self.image_size),
        }
        return image_t, targets


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: (torch.stack([b[0] for b in batch], dim=0), [b[1] for b in batch]),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )
