from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
import json

import numpy as np

import torch
import yaml

LOCAL = Path(__file__).resolve().parent
ROOT_2D = LOCAL.parent / "lung_detection_2D_one_slide"

# Force the 3D prototype to use its local modules even when the parent project
# is also on PYTHONPATH / sys.path.
for p in (str(ROOT_2D), str(LOCAL)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dataloader_mod = _load_module("lidc3d_dataloader", LOCAL / "dataloader.py")
_model_mod = _load_module("lidc3d_model", LOCAL / "model.py")
_losses_mod = _load_module("lidc3d_losses", ROOT_2D / "losses.py")
_utils_mod = _load_module("lidc3d_utils", ROOT_2D / "utils.py")

LIDC3DContextDataset = _dataloader_mod.LIDC3DContextDataset
build_loader = _dataloader_mod.build_loader
Yolo3DContext = _model_mod.Yolo3DContext
YoloLoss = _losses_mod.YoloLoss
decode_predictions = _losses_mod.decode_predictions
compute_map50 = _utils_mod.compute_map50
nms = _utils_mod.nms


def _prepare_anchors(cfg):
    anchors = cfg["model"].get("anchors")
    if anchors:
        return torch.tensor(anchors, dtype=torch.float32)
    return torch.tensor(cfg["model"]["anchors_fallback"], dtype=torch.float32)


def _post_process(preds, anchors, strides, cfg):
    dec = decode_predictions(preds, anchors, strides, conf_thr=cfg["infer"]["conf_threshold"])
    out = []
    for d in dec:
        if d["boxes"].shape[0] == 0:
            out.append({"boxes": d["boxes"], "scores": d["scores"], "mal_probs": d["mal_probs"]})
            continue
        keep = nms(d["boxes"], d["scores"], iou_thr=cfg["infer"]["iou_threshold"])
        keep = keep[: cfg["infer"]["max_det"]]
        out.append({
            "boxes": d["boxes"][keep],
            "scores": d["scores"][keep],
            "mal_probs": d["mal_probs"][keep],
        })
    return out


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    anchors = _prepare_anchors(cfg).to(device)
    all_pred, all_tgt, mal_pred, mal_true = [], [], [], []
    for x, targets in loader:
        x = x.to(device)
        preds = model(x)
        dets = _post_process(preds, anchors, cfg["model"]["strides"], cfg)
        for det, tgt in zip(dets, targets):
            all_pred.append({"boxes": det["boxes"].cpu().numpy(), "scores": det["scores"].cpu().numpy()})
            all_tgt.append({"boxes": tgt["boxes"].cpu().numpy()})
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
    metrics = {"mAP50": float(compute_map50(all_pred, all_tgt))}
    if mal_pred:
        mp = np.asarray(mal_pred)
        mt = np.asarray(mal_true)
        metrics["mal_accuracy"] = float(((mp >= 0.5).astype(int) == mt).mean())
        try:
            from sklearn.metrics import roc_auc_score, confusion_matrix
            if len(np.unique(mt)) == 2:
                metrics["mal_auc"] = float(roc_auc_score(mt, mp))
            cm = confusion_matrix(mt, (mp >= 0.5).astype(int), labels=[0, 1])
            metrics["mal_confusion"] = cm.tolist()
        except Exception:
            pass
    return metrics


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = LIDC3DContextDataset(
        cfg["data"]["train_csv"],
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
    loader = build_loader(ds_train, args.batch_size, True, args.num_workers)

    model = Yolo3DContext(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = YoloLoss(
        anchors=cfg["model"]["anchors_fallback"],
        strides=cfg["model"]["strides"],
        box_weight=cfg["loss"]["box_weight"],
        obj_weight=cfg["loss"]["obj_weight"],
        mal_weight=cfg["loss"]["mal_weight"],
        obj_pos_weight=cfg["loss"]["obj_pos_weight"],
        mal_pos_weight=cfg["loss"]["mal_pos_weight"],
        image_size=cfg["data"]["image_size"],
    ).to(device)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, targets in loader:
            x = x.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["malignancy"] = t["malignancy"].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss, _ = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.6f}")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, save_dir / "3d_context_best.pt")
    print(f"saved to {save_dir / '3d_context_best.pt'}")


def test(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = LIDC3DContextDataset(
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
    loader = build_loader(ds_test, args.batch_size, False, args.num_workers)

    model = Yolo3DContext(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)

    if args.weights:
        ckpt_path = Path(args.weights)
        print(f"loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)

    metrics = evaluate(model, loader, device, cfg)
    print(json.dumps(metrics, indent=2))


def predict(cfg: dict, args):
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
    loader = build_loader(ds, args.batch_size, False, args.num_workers)
    model = Yolo3DContext(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)
    ckpt = torch.load(args.weights, map_location=device) if args.weights else None
    if ckpt is not None:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device)
            preds = model(x)
            anchors = _prepare_anchors(cfg).to(device)
            dets = _post_process(preds, anchors, cfg["model"]["strides"], cfg)
            for det, tgt in zip(dets, targets):
                img_id = str(tgt["image_id"])
                patient_id = img_id.rsplit("_", 1)[0]
                slice_id = int(img_id.rsplit("_", 1)[1])
                for box, conf, mal in zip(det["boxes"].cpu().tolist(), det["scores"].cpu().tolist(), det["mal_probs"].cpu().tolist()):
                    rows.append({
                        "patient_id": patient_id,
                        "slice": slice_id,
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "conf": float(conf),
                        "malignancy_prob": float(mal),
                    })
    with out_path.open("w", newline="") as f:
        writer = __import__("csv").writer(f)
        writer.writerow(["patient_id", "slice", "x1", "y1", "x2", "y2", "conf", "malignancy_prob"])
        for r in rows:
            writer.writerow([r["patient_id"], r["slice"], r["x1"], r["y1"], r["x2"], r["y2"], r["conf"], r["malignancy_prob"]])
    print(f"wrote {len(rows)} detections to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--weights", type=str, default="checkpoints/3d_context_best.pt")
    ap.add_argument("--output", type=str, default="detections_out/detections_3d_context.csv")
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg, args)
    elif args.predict:
        predict(cfg, args)
    else:
        test(cfg, args)


if __name__ == "__main__":
    main()
