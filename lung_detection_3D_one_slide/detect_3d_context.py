from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROOT_PROJ = ROOT / "lung_detection_2D_one_slide"
if str(ROOT_PROJ) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJ))

from losses import decode_predictions
from utils import nms
from dataloader import LIDC3DContextDataset
from model import Yolo3DContext


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LIDC3DContextDataset(
        cfg["data"]["test_csv"],
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        image_size=cfg["data"]["image_size"],
        depth=cfg["data"].get("volume_depth", 3),
        col_patient=cfg["data"]["col_patient"],
        col_slice=cfg["data"]["col_center_slice"],
        col_malignancy=cfg["data"]["col_malignancy"],
        col_nodule_idxs=cfg["data"]["col_nodule_idxs"],
        col_image_path=cfg["data"]["col_image_path"],
        col_mask_path=cfg["data"]["col_mask_path"],
        in_channels=cfg["data"]["in_channels"],
    )

    model = Yolo3DContext(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)

    ckpt = torch.load("checkpoints/3d_context_best.pt", map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    anchors = torch.tensor(cfg["model"]["anchors_fallback"], dtype=torch.float32, device=device)
    anchors = anchors.permute(0, 2, 1).reshape(3, 3, 2)
    out_dir = Path("detections_out")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "detections_3d_context.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "slice", "x1", "y1", "x2", "y2", "conf", "malignancy_prob"])
        for idx in range(len(ds)):
            x, target = ds[idx]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(x)
            dec = decode_predictions(preds, anchors, cfg["model"]["strides"], conf_thr=cfg["infer"]["conf_threshold"])[0]
            if dec["boxes"].numel() > 0:
                keep = nms(dec["boxes"], dec["scores"], iou_thr=cfg["infer"]["iou_threshold"])
                keep = keep[: cfg["infer"]["max_det"]]
                boxes = dec["boxes"][keep].cpu().numpy()
                scores = dec["scores"][keep].cpu().numpy()
                mals = dec["mal_probs"][keep].cpu().numpy()
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                mals = np.zeros((0,), dtype=np.float32)

            patient_id = str(target["image_id"]).split("_")[0]
            slice_id = int(str(target["image_id"]).split("_")[-1]) if "_" in str(target["image_id"]) else 0
            for (x1, y1, x2, y2), sc, mp in zip(boxes, scores, mals):
                writer.writerow([patient_id, slice_id, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{sc:.4f}", f"{mp:.4f}"])

    print(f"saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
