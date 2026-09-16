from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from dataloader import _center_mask_2d, _to_list
from utils import _iou_np, bboxes_from_mask_multi


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_key(patient_id: str, slice_idx: int) -> str:
    return f"{patient_id}__{int(slice_idx)}"


def parse_predictions(csv_path: Path, min_conf: float = 0.0) -> Dict[str, List[dict]]:
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns or "slice" not in df.columns:
        raise ValueError("Prediction CSV must contain patient_id and slice columns")
    preds: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        if float(row.get("conf", 0.0)) < min_conf:
            continue
        key = sample_key(str(row["patient_id"]), int(row["slice"]))
        entry = {
            "box": np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32),
            "score": float(row.get("conf", 0.0)),
            "mal_prob": float(row.get("malignancy_prob", 0.0)) if "malignancy_prob" in row else None,
        }
        preds.setdefault(key, []).append(entry)
    return preds


def load_ground_truth(csv_path: Path, cfg: dict) -> Dict[str, dict]:
    df = pd.read_csv(csv_path)
    expected_cols = [cfg["data"]["col_patient"], cfg["data"]["col_center_slice"]]
    if any(col not in df.columns for col in expected_cols):
        raise ValueError(f"Ground truth CSV missing required columns: {expected_cols}")

    gts: Dict[str, dict] = {}
    for _, row in df.iterrows():
        patient_id = str(row[cfg["data"]["col_patient"]])
        slice_idx = int(row[cfg["data"]["col_center_slice"]])
        key = sample_key(patient_id, slice_idx)
        gts[key] = {
            "patient_id": patient_id,
            "slice": slice_idx,
            "image_path": row.get(cfg["data"]["col_image_path"], None),
            "mask_path": row.get(cfg["data"]["col_mask_path"], None),
            "malignancy": _to_list(row[cfg["data"]["col_malignancy"]]) if cfg["data"]["col_malignancy"] in row else [],
        }
    return gts


def resolve_mask_path(sample: dict, cfg: dict) -> Path:
    mask_path = sample.get("mask_path")
    if isinstance(mask_path, str) and mask_path:
        p = Path(mask_path)
        if p.exists():
            return p
    images_dir = Path(cfg["data"]["masks_dir"])
    patient = sample["patient_id"]
    slice_idx = sample["slice"]
    guess = images_dir / f"{patient}_slice_{slice_idx}_mask.npy"
    if guess.exists():
        return guess
    raise FileNotFoundError(f"Mask not found for sample {patient} slice {slice_idx}")


def mask_to_boxes(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path)
    mask2d = _center_mask_2d(mask)
    boxes = bboxes_from_mask_multi(mask2d, min_area=1)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def match_sample(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    iou_thr: float,
) -> Tuple[int, int, int, List[int]]:
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    matches = []
    tp = 0
    for i, box in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            matches.append(-1)
            continue
        ious = _iou_np(box[None, :], gt_boxes)[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not matched_gt[j]:
            matched_gt[j] = True
            tp += 1
            matches.append(j)
        else:
            matches.append(-1)
    fp = len(pred_boxes) - tp
    fn = int(np.count_nonzero(~matched_gt))
    return tp, fp, fn, matches


def score_malignancy(
    gt_mals: List[int],
    pred_mals: np.ndarray,
    matches: List[int],
) -> Tuple[int, int, int]:
    if pred_mals is None or len(gt_mals) == 0:
        return 0, 0, 0
    correct = 0
    total = 0
    for pred_idx, gt_idx in enumerate(matches):
        if gt_idx < 0:
            continue
        pred_label = int(pred_mals[pred_idx] >= 0.5)
        gt_label = int(gt_mals[gt_idx]) if gt_idx < len(gt_mals) else 0
        correct += int(pred_label == gt_label)
        total += 1
    return correct, total, len(gt_mals)


def summarize_metrics(total_tp: int, total_fp: int, total_fn: int) -> dict:
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection CSV against ground-truth masks.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file path")
    parser.add_argument("--gt-csv", default=None, help="Ground-truth CSV path (overrides config test_csv)")
    parser.add_argument("--pred-csv", default="detections_out/detections.csv", help="Prediction CSV path")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for true positives")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence filter for predictions")
    parser.add_argument("--patient-id", default=None, help="Evaluate only this patient_id")
    parser.add_argument("--slice", type=int, default=None, help="Evaluate only this slice number")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gt_csv = Path(args.gt_csv if args.gt_csv is not None else cfg["data"]["test_csv"])
    if not gt_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_csv}")
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    preds = parse_predictions(pred_csv, min_conf=args.min_conf)
    gts = load_ground_truth(gt_csv, cfg)

    samples = sorted(preds.keys())
    if args.patient_id is not None and args.slice is not None:
        samples = [sample_key(args.patient_id, args.slice)]
    elif args.patient_id is not None:
        samples = [k for k in preds if k.startswith(f"{args.patient_id}__")]
    elif args.slice is not None:
        samples = [k for k in preds if k.endswith(f"__{args.slice}")]

    if not samples:
        raise ValueError("No prediction samples found for evaluation")

    report = []
    total_tp = total_fp = total_fn = 0
    total_mal_correct = total_mal_total = 0
    total_gt_boxes = 0

    for key in samples:
        if key not in gts:
            print(f"WARNING: no ground truth row for sample {key}")
            continue
        gt = gts[key]
        mask_path = resolve_mask_path(gt, cfg)
        gt_boxes = mask_to_boxes(mask_path)
        pred_entries = preds.get(key, [])
        pred_boxes = np.stack([p["box"] for p in pred_entries], axis=0) if pred_entries else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.asarray([p["score"] for p in pred_entries], dtype=np.float32)
        pred_mals = np.asarray([p["mal_prob"] for p in pred_entries], dtype=np.float32) if any(p["mal_prob"] is not None for p in pred_entries) else None

        tp, fp, fn, matches = match_sample(gt_boxes, pred_boxes, pred_scores, args.iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_boxes += len(gt_boxes)
        mal_correct, mal_total, _ = score_malignancy(_to_list(gt["malignancy"]), pred_mals, matches)
        total_mal_correct += mal_correct
        total_mal_total += mal_total

        report.append({
            "sample": key,
            "gt_boxes": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mal_matched": mal_total,
            "mal_correct": mal_correct,
            "mask_path": str(mask_path),
        })

    metrics = summarize_metrics(total_tp, total_fp, total_fn)
    if total_mal_total > 0:
        metrics["malignancy_accuracy"] = total_mal_correct / total_mal_total

    output = {
        "config": str(args.config),
        "gt_csv": str(gt_csv),
        "pred_csv": str(pred_csv),
        "iou_threshold": args.iou,
        "min_conf": args.min_conf,
        "samples_evaluated": len(report),
        "total_gt_boxes": total_gt_boxes,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "metrics": metrics,
        "details": report,
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"Evaluated {len(report)} samples")
        print(f"GT boxes: {total_gt_boxes} | TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        if "malignancy_accuracy" in metrics:
            print(f"Malignancy accuracy (matched detections): {metrics['malignancy_accuracy']:.4f}")
        print("---")
        for item in report:
            print(
                f"{item['sample']} | GT={item['gt_boxes']} | PRED={item['pred_boxes']} | "
                f"TP={item['tp']} FP={item['fp']} FN={item['fn']} | mal_correct={item['mal_correct']}/{item['mal_matched']}"
            )


if __name__ == "__main__":
    main()
