from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml

from losses import decode_predictions
from model import YoloLIDC
from utils import draw_detections, nms


# --------------------------- Preprocessing --------------------------------
HU_MIN, HU_MAX = -1000, 400


def _window_normalize(vol: np.ndarray, hu_min: int = HU_MIN, hu_max: int = HU_MAX) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max).astype(np.float32)
    return (vol - hu_min) / (hu_max - hu_min)


def _resize_2d(sl: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(sl, (size, size), interpolation=cv2.INTER_LINEAR)


def _load_dicom_series(folder: str) -> Tuple[np.ndarray, str]:
    import pydicom
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {folder}")
    slices = [pydicom.dcmread(f) for f in files]
    try:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    except Exception:
        pass
    imgs = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(s, "RescaleIntercept", 0.0) or 0.0)
        imgs.append(arr * slope + intercept)  # HU
    vol = np.stack(imgs, axis=0)              # (Z,H,W)
    pid = getattr(slices[0], "PatientID", Path(folder).name)
    return vol, str(pid)


def _load_nifti(path: str) -> Tuple[np.ndarray, str]:
    import nibabel as nib
    img = nib.load(path)
    vol = np.asanyarray(img.dataobj).astype(np.float32)      # (H,W,Z) usually
    # move Z axis to first for consistency
    if vol.ndim == 3 and vol.shape[-1] < vol.shape[0]:
        vol = np.transpose(vol, (2, 0, 1))
    return vol, Path(path).stem


def _stack_25d(vol_norm: np.ndarray, z: int) -> np.ndarray:
    """Return (3,H,W) slice using z-1, z, z+1 (edges duplicated)."""
    Z = vol_norm.shape[0]
    z0 = max(0, z - 1)
    z2 = min(Z - 1, z + 1)
    return np.stack([vol_norm[z0], vol_norm[z], vol_norm[z2]], axis=0)


# --------------------------- Model / postproc -----------------------------
def _load_model(weights: str, device: torch.device):
    ckpt = torch.load(weights, map_location=device)
    cfg = ckpt["cfg"]
    model = YoloLIDC(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    anchors = torch.tensor(ckpt["anchors"], dtype=torch.float32, device=device)  # (S,A,2)
    return model, anchors, cfg


def _detect_slice(
    model, anchors, cfg, tensor_chw: np.ndarray, device, conf_thr: float, iou_thr: float,
):
    x = torch.from_numpy(tensor_chw).float().unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)
    dec = decode_predictions(preds, anchors, cfg["model"]["strides"], conf_thr=conf_thr)[0]
    if dec["boxes"].shape[0] == 0:
        return dec
    keep = nms(dec["boxes"], dec["scores"], iou_thr=iou_thr)
    return {
        "boxes": dec["boxes"][keep].cpu().numpy(),
        "scores": dec["scores"][keep].cpu().numpy(),
        "mal_probs": dec["mal_probs"][keep].cpu().numpy(),
    }


# --------------------------- Main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help=".npy | .dcm folder | .nii/.nii.gz")
    ap.add_argument("--out", default="detections_out")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--stride_z", type=int, default=1, help="run every Nth slice in mode B")
    ap.add_argument("--patient_id", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, anchors, cfg = _load_model(args.weights, device)
    size = cfg["data"]["image_size"]
    conf_thr = args.conf if args.conf is not None else cfg["infer"]["conf_threshold"]
    iou_thr = args.iou if args.iou is not None else cfg["infer"]["iou_threshold"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "detections.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["patient_id", "slice", "x1", "y1", "x2", "y2", "conf", "malignancy_prob"])

    inp = Path(args.input)
    # ---------------- Mode A ----------------
    if inp.suffix == ".npy":
        arr = np.load(inp)
        # normalise to (3,H,W)
        from dataloader import _to_chw_3
        tensor = _to_chw_3(arr)
        if tensor.max() > 1.5:
            tensor = tensor / tensor.max()
        tensor = np.clip(tensor, 0, 1)
        if tensor.shape[1] != size or tensor.shape[2] != size:
            tensor = np.stack([_resize_2d(c, size) for c in tensor], axis=0)
        dets = _detect_slice(model, anchors, cfg, tensor.astype(np.float32), device, conf_thr, iou_thr)
        pid = args.patient_id or inp.stem
        vis = (np.transpose(tensor, (1, 2, 0)) * 255).astype(np.uint8)
        annotated = draw_detections(
            vis, dets["boxes"] if isinstance(dets.get("boxes"), np.ndarray) else np.zeros((0, 4)),
            dets["scores"] if isinstance(dets.get("scores"), np.ndarray) else None,
            dets["mal_probs"] if isinstance(dets.get("mal_probs"), np.ndarray) else None,
        )
        cv2.imwrite(str(out_dir / f"{pid}.png"), annotated)
        if isinstance(dets.get("boxes"), np.ndarray):
            for (x1, y1, x2, y2), sc, mp in zip(dets["boxes"], dets["scores"], dets["mal_probs"]):
                writer.writerow([pid, 0, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                                 f"{sc:.4f}", f"{mp:.4f}"])
        csv_f.close()
        print(f"wrote {out_dir/(pid+'.png')} and {csv_path}")
        return

    # ---------------- Mode B ----------------
    if inp.is_dir():
        vol, pid = _load_dicom_series(str(inp))
    elif inp.name.endswith(".nii") or inp.name.endswith(".nii.gz"):
        vol, pid = _load_nifti(str(inp))
    else:
        raise ValueError(f"unsupported input {inp}")
    pid = args.patient_id or pid

    vol_norm = _window_normalize(vol)                          # (Z,H,W) in [0,1]
    # resize to model resolution
    Z = vol_norm.shape[0]
    vol_r = np.stack([_resize_2d(vol_norm[z], size) for z in range(Z)], axis=0)

    all_dets = []
    for z in range(0, Z, args.stride_z):
        chw = _stack_25d(vol_r, z).astype(np.float32)
        dets = _detect_slice(model, anchors, cfg, chw, device, conf_thr, iou_thr)
        if isinstance(dets.get("boxes"), np.ndarray) and len(dets["boxes"]):
            vis = (vol_r[z] * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            annotated = draw_detections(vis, dets["boxes"], dets["scores"], dets["mal_probs"])
            cv2.imwrite(str(out_dir / f"{pid}_z{z:04d}.png"), annotated)
            for (x1, y1, x2, y2), sc, mp in zip(dets["boxes"], dets["scores"], dets["mal_probs"]):
                writer.writerow([pid, z, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                                 f"{sc:.4f}", f"{mp:.4f}"])
                all_dets.append((z, [x1, y1, x2, y2], float(sc), float(mp)))

    # optional 3D NMS aggregation across Z (simple 2D-IoU per adjacent slice)
    csv_f.close()
    print(f"processed {Z} slices | {sum(1 for _ in all_dets)} detections | csv -> {csv_path}")


if __name__ == "__main__":
    main()