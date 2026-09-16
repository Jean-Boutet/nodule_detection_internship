from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from dataloader import _center_mask_2d, _to_single_center_slice
from segmentation import UNet


def load_guidance_mapping(path: str) -> Dict[str, List[List[int]]]:
    with open(path, "r") as f:
        data = json.load(f)
    return {str(k): [list(map(int, box)) for box in boxes] for k, boxes in data.items()}


def build_guidance_mask(
    shape: Sequence[int],
    boxes: Optional[List[Sequence[int]]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if boxes is None:
        return mask
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = 1
    return mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return np.clip(image, 0.0, 1.0)


def make_visualization(
    image: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: Optional[np.ndarray] = None,
) -> np.ndarray:
    image_vis = (normalize_image(image) * 255.0).astype(np.uint8)
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_GRAY2BGR)

    pred_vis = (mask_pred * 255.0).astype(np.uint8)
    pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
    if mask_gt is not None:
        gt_vis = (mask_gt.astype(np.uint8) * 255.0).astype(np.uint8)
        gt_vis = cv2.applyColorMap(gt_vis, cv2.COLORMAP_JET)
        output = np.concatenate([image_vis, pred_vis, gt_vis], axis=1)
    else:
        output = np.concatenate([image_vis, pred_vis], axis=1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict segmentation on a single numpy CT patch")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input-npy", type=str, required=True)
    parser.add_argument("--output-image", type=str, default="segmentation_prediction.png")
    parser.add_argument("--mask-path", type=str, default=None)
    parser.add_argument("--guidance-json", type=str, default=None)
    parser.add_argument("--guidance-key", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = load_checkpoint(args.weights, device)
    model_args = checkpoint.get("args", {})
    in_channels = 2 if model_args.get("with_guidance", False) else 1
    model = UNet(in_channels=in_channels, out_channels=1)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    image = np.load(args.input_npy)
    image = _to_single_center_slice(image)

    guidance_boxes = None
    if args.guidance_json is not None:
        guidance_mapping = load_guidance_mapping(args.guidance_json)
        key = args.guidance_key if args.guidance_key is not None else Path(args.input_npy).name
        guidance_boxes = guidance_mapping.get(str(key))

    if guidance_boxes is not None:
        guidance_mask = build_guidance_mask(image.shape, guidance_boxes)
        image_tensor = np.stack([image, guidance_mask.astype(np.float32)], axis=0)
    else:
        image_tensor = np.expand_dims(image.astype(np.float32), axis=0)

    tensor = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs >= args.threshold).astype(np.uint8)

    gt_mask = None
    if args.mask_path is not None:
        mask = np.load(args.mask_path)
        gt_mask = _center_mask_2d(mask).astype(np.uint8)
        if gt_mask.shape != image.shape:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    output = make_visualization(image, pred_mask, gt_mask)
    out_path = Path(args.output_image).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), output)
    print(f"Saved prediction visualization to {out_path}")


if __name__ == "__main__":
    main()
