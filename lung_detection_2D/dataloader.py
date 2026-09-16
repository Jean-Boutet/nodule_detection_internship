from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import bboxes_from_mask_multi


# --------------------------------------------------------------------------
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


def _find_file(root: Path, patient: str, center_slice: int) -> Path | None:
    stem = f"{patient}_{int(center_slice)}"
    p = root / f"{stem}.npy"
    if p.exists():
        return p
    p = root / patient / f"{int(center_slice)}.npy"
    if p.exists():
        return p
    hits = list(root.rglob(f"{stem}*.npy"))
    return hits[0] if hits else None


def _to_chw_3(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary shape into (3,H,W) float32."""
    a = np.asarray(arr)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=0)
    elif a.ndim == 3:
        if a.shape[-1] in (1, 3):        # HWC -> CHW
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            a = np.transpose(a, (2, 0, 1))
        elif a.shape[0] in (1, 3):       # already CHW
            if a.shape[0] == 1:
                a = np.repeat(a, 3, axis=0)
        else:
            raise ValueError(f"Unsupported image shape {a.shape}")
    else:
        raise ValueError(f"Unsupported image shape {a.shape}")
    return a.astype(np.float32)


def _center_mask_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        return (a > 0).astype(np.uint8)
    if a.ndim == 3:
        if a.shape[-1] in (1, 3):
            mid = a.shape[-1] // 2
            return (a[..., mid] > 0).astype(np.uint8)
        if a.shape[0] in (1, 3):
            mid = a.shape[0] // 2
            return (a[mid] > 0).astype(np.uint8)
    raise ValueError(f"Unsupported mask shape {a.shape}")


def _to_single_center_slice(arr: np.ndarray) -> np.ndarray:
    """Extract the central slice from a (3,H,W) or (H,W,3) patch."""
    a = np.asarray(arr)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        if a.shape[0] == 3:
            return a[1].astype(np.float32)
        if a.shape[-1] == 3:
            return a[..., 1].astype(np.float32)
    raise ValueError(f"Unsupported image shape {a.shape}")


# --------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    """Load image and mask pairs for 2D central-slice segmentation."""

    def __init__(
        self,
        csv_path: str | os.PathLike,
        images_dir: str | os.PathLike,
        masks_dir: str | os.PathLike,
        col_image_path: str = "image_path",
        col_mask_path: str = "mask_path",
        image_size: Optional[int] = None,
        augment: bool = False,
        aug_cfg: Optional[Dict[str, Any]] = None,
        with_guidance: bool = False,
        guidance_mapping: Optional[Dict[str, Sequence[Tuple[int, int, int, int]]]] = None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.col_image_path = col_image_path
        self.col_mask_path = col_mask_path
        self.image_size = image_size
        self.augment = augment
        self.with_guidance = with_guidance
        self.guidance_mapping = guidance_mapping or {}
        aug_cfg = aug_cfg or {}

        if augment:
            if A is None:
                raise ImportError(
                    "albumentations is required for augmentation. Install it or set augment=False."
                )
            self.tf = A.Compose(
                [
                    A.HorizontalFlip(p=aug_cfg.get("hflip", 0.5)),
                    A.VerticalFlip(p=aug_cfg.get("vflip", 0.5)),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.10,
                        rotate_limit=aug_cfg.get("rotate_limit", 10),
                        border_mode=0,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=aug_cfg.get("brightness_limit", 0.15),
                        contrast_limit=aug_cfg.get("contrast_limit", 0.15),
                        p=0.5,
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=0.20,
                    ),
                    A.GaussNoise(
                        std_range=(0.01, 0.05),
                        p=0.20,
                    ),
                ]
            )
        else:
            self.tf = None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, root: Path, row: pd.Series, col_name: str) -> Path:
        if col_name in row and isinstance(row[col_name], str):
            p = Path(row[col_name])
            if not p.is_absolute():
                p = root / p
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not resolve path for {col_name}: {row[col_name]}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(self.images_dir, row, self.col_image_path)
        mask_path = self._resolve_path(self.masks_dir, row, self.col_mask_path)

        image = _to_single_center_slice(np.load(img_path))
        mask = _center_mask_2d(np.load(mask_path)).astype(np.uint8)

        guidance_mask = None
        if self.with_guidance:
            guidance_mask = np.zeros_like(mask, dtype=np.uint8)
            key = str(row.get(self.col_image_path, ""))
            guidance_boxes = self.guidance_mapping.get(key)
            if guidance_boxes is None:
                guidance_boxes = self.guidance_mapping.get(img_path.name)
            if guidance_boxes is not None:
                for x1, y1, x2, y2 in guidance_boxes:
                    guidance_mask[int(y1):int(y2), int(x1):int(x2)] = 1

        if self.image_size is not None and (image.shape[0] != self.image_size or image.shape[1] != self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            if guidance_mask is not None:
                guidance_mask = cv2.resize(guidance_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.tf is not None:
            aug = self.tf(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims((mask > 0).astype(np.float32), axis=0)

        if guidance_mask is not None:
            guidance_mask = np.expand_dims((guidance_mask > 0).astype(np.float32), axis=0)
            image = np.concatenate([image, guidance_mask], axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)


# --------------------------------------------------------------------------
class LIDCDetectionDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
    images_dir : str    directory containing .npy tensors
    masks_dir  : str    directory containing .npy masks
    image_size : int    expected spatial size (used for augmentation)
    augment    : bool   apply training augmentations
    col_*      : column overrides
    """

    def __init__(
        self,
        csv_path: str | os.PathLike,
        images_dir: str | os.PathLike,
        masks_dir: str | os.PathLike,
        image_size: int = 256,
        augment: bool = False,
        col_patient: str = "patient_id",
        col_slice: str = "center_slice",
        col_malignancy: str = "malignancy_labels",
        col_nodule_idxs: str = "grouped_nodule_idxs",
        col_image_path: str = "image_path",
        col_mask_path: str = "mask_path",
        aug_cfg: Dict[str, Any] | None = None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.col_patient = col_patient
        self.col_slice = col_slice
        self.col_malignancy = col_malignancy
        self.col_nodule_idxs = col_nodule_idxs
        self.col_image_path = col_image_path
        self.col_mask_path = col_mask_path

        aug_cfg = aug_cfg or {}
        if augment:
            if A is None:
                raise ImportError(
                    "albumentations is required for augmentation. Install it or set augment=False."
                )
            self.tf = A.Compose(
                [
                    A.HorizontalFlip(p=aug_cfg.get("hflip", 0.5)),

                    A.VerticalFlip(p=aug_cfg.get("vflip", 0.5)),

                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.10,
                        rotate_limit=aug_cfg.get("rotate_limit", 10),
                        border_mode=0,
                        p=0.5,
                    ),

                    A.RandomBrightnessContrast(
                        brightness_limit=aug_cfg.get("brightness_limit", 0.15),
                        contrast_limit=aug_cfg.get("contrast_limit", 0.15),
                        p=0.5,
                    ),

                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=0.20,
                    ),

                    A.GaussNoise(
                        std_range=(0.01, 0.05),
                        p=0.20,
                    ),

                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=0.20,
                    ),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels", "malignancy"],
                    min_area=1,
                    min_visibility=0.1,
                ),
            )
        else:
            self.tf = None

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def _resolve_paths(self, row) -> Tuple[Path, Path]:
        img_path = None
        mask_path = None
        if self.col_image_path in row and isinstance(row[self.col_image_path], str):
            p = Path(row[self.col_image_path])
            if not p.is_absolute():
                p = self.images_dir / p
            if p.exists():
                img_path = p
        if self.col_mask_path in row and isinstance(row[self.col_mask_path], str):
            p = Path(row[self.col_mask_path])
            if not p.is_absolute():
                p = self.masks_dir / p
            if p.exists():
                mask_path = p
        if img_path is None:
            img_path = _find_file(self.images_dir, str(row[self.col_patient]), int(row[self.col_slice]))
        if mask_path is None:
            mask_path = _find_file(self.masks_dir, str(row[self.col_patient]), int(row[self.col_slice]))
        if img_path is None or mask_path is None:
            raise FileNotFoundError(
                f"Could not resolve image/mask for patient={row[self.col_patient]} "
                f"slice={row[self.col_slice]}"
            )
        return img_path, mask_path

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path, mask_path = self._resolve_paths(row)

        image = _to_chw_3(np.load(img_path))            # (3,H,W) float32
        mask2d = _center_mask_2d(np.load(mask_path))    # (H,W) uint8

        # bboxes ranked by area (largest first) for alignment with malignancy list
        bxs = bboxes_from_mask_multi(mask2d, min_area=4)
        bxs.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

        mal_all = [int(v) for v in _to_list(row[self.col_malignancy])] if self.col_malignancy in self.df.columns else []
        if mal_all and len(mal_all) != len(bxs):
            mal_all = (mal_all + [0] * len(bxs))[: len(bxs)]
        elif not mal_all:
            mal_all = [0] * len(bxs)

        boxes = np.asarray(bxs, dtype=np.float32).reshape(-1, 4)
        malignancy = np.asarray(mal_all, dtype=np.float32).reshape(-1)
        labels = np.zeros((len(boxes),), dtype=np.int64)

        # min/max clip intensity per channel to [0,1] (already preprocessed but safe)
        if image.max() > 1.5:
            image = image / max(image.max(), 1e-6)
        image = np.clip(image, 0.0, 1.0)

        # albumentations expects HWC uint8/float in [0,1]
        if self.tf is not None:
            img_hwc = np.transpose(image, (1, 2, 0))
            out = self.tf(
                image=img_hwc,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist(),
                malignancy=malignancy.tolist(),
            )
            img_hwc = np.asarray(out["image"], dtype=np.float32)
            image = np.transpose(img_hwc, (2, 0, 1))
            boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(out["class_labels"], dtype=np.int64).reshape(-1)
            malignancy = np.asarray(out["malignancy"], dtype=np.float32).reshape(-1)

        image_t = torch.from_numpy(np.ascontiguousarray(image))
        targets: Dict[str, Any] = {
            "boxes": torch.from_numpy(boxes) if len(boxes) else torch.zeros((0, 4)),
            "labels": torch.from_numpy(labels) if len(labels) else torch.zeros((0,), dtype=torch.long),
            "malignancy": torch.from_numpy(malignancy) if len(malignancy) else torch.zeros((0,)),
            "image_id": f"{row[self.col_patient]}_{int(row[self.col_slice])}",
            "orig_size": (image.shape[1], image.shape[2]),
        }
        return image_t, targets


# --------------------------------------------------------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


def build_loader(
    dataset: LIDCDetectionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )
