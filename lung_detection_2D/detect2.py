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
from utils import draw_detections, nms, bboxes_from_mask_multi
from dataloader import _center_mask_2d


# --------------------------- Preprocessing --------------------------------

HU_MIN, HU_MAX = -1000, 400


def _window_normalize(
    vol: np.ndarray,
    hu_min: int = HU_MIN,
    hu_max: int = HU_MAX
) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max).astype(np.float32)
    return (vol - hu_min) / (hu_max - hu_min)


def _resize_2d(sl: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(
        sl,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )


def _load_dicom_series(folder: str) -> Tuple[np.ndarray, str]:
    import pydicom

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))

    if not files:
        raise FileNotFoundError(
            f"No DICOM files in {folder}"
        )

    slices = [pydicom.dcmread(f) for f in files]

    try:
        slices.sort(
            key=lambda s: int(
                getattr(s, "InstanceNumber", 0)
            )
        )
    except Exception:
        pass

    imgs = []

    for s in slices:
        arr = s.pixel_array.astype(np.float32)

        slope = float(
            getattr(s, "RescaleSlope", 1.0) or 1.0
        )

        intercept = float(
            getattr(s, "RescaleIntercept", 0.0) or 0.0
        )

        imgs.append(
            arr * slope + intercept
        )

    vol = np.stack(imgs, axis=0)

    pid = getattr(
        slices[0],
        "PatientID",
        Path(folder).name
    )

    return vol, str(pid)


def _load_nifti(path: str) -> Tuple[np.ndarray, str]:
    import nibabel as nib

    img = nib.load(path)

    vol = np.asanyarray(
        img.dataobj
    ).astype(np.float32)

    # Move Z axis to first for consistency
    if vol.ndim == 3 and vol.shape[-1] < vol.shape[0]:
        vol = np.transpose(
            vol,
            (2, 0, 1)
        )

    return vol, Path(path).stem


def _stack_25d(
    vol_norm: np.ndarray,
    z: int
) -> np.ndarray:
    """
    Return (3,H,W) using:
        z-1
        z
        z+1

    At the volume edges, duplicate the closest slice.
    """

    Z = vol_norm.shape[0]

    z0 = max(0, z - 1)
    z2 = min(Z - 1, z + 1)

    return np.stack(
        [
            vol_norm[z0],
            vol_norm[z],
            vol_norm[z2]
        ],
        axis=0
    )


# --------------------------- Ground truth ---------------------------------


def _prepare_ground_truth_mask(
    mask_path: str,
    size: int
) -> np.ndarray:
    """
    Load the ground-truth mask and convert it to a 2D mask
    compatible with the model input resolution.
    """

    mask = np.load(mask_path)

    # Convert possible 3D / multi-channel mask to the
    # center 2D mask used for the detection task.
    mask2d = _center_mask_2d(mask)

    # Ensure numpy array
    mask2d = np.asarray(mask2d)

    # Resize mask if necessary.
    # IMPORTANT: nearest-neighbor interpolation is used
    # because this is a binary/label mask.
    if mask2d.shape[0] != size or mask2d.shape[1] != size:
        mask2d = cv2.resize(
            mask2d.astype(np.uint8),
            (size, size),
            interpolation=cv2.INTER_NEAREST
        )

    return mask2d


def _draw_ground_truth(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Draw ground-truth bounding boxes extracted from the mask.
    """

    output = image.copy()

    gt_boxes = bboxes_from_mask_multi(
        mask,
        min_area=1
    )

    for box in gt_boxes:
        x1, y1, x2, y2 = box

        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

    return output


def _add_title(
    image: np.ndarray,
    title: str
) -> np.ndarray:
    """
    Add a title at the top of an image.
    """

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# --------------------------- Model / postproc -----------------------------


def _load_model(
    weights: str,
    device: torch.device
):
    ckpt = torch.load(
        weights,
        map_location=device
    )

    cfg = ckpt["cfg"]

    model = YoloLIDC(
        in_channels=cfg["data"]["in_channels"],
        backbone=cfg["model"]["backbone"],
        num_anchors_per_scale=cfg["model"]["num_anchors_per_scale"],
        strides=cfg["model"]["strides"],
    ).to(device)

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    anchors = torch.tensor(
        ckpt["anchors"],
        dtype=torch.float32,
        device=device
    )

    return model, anchors, cfg


def _detect_slice(
    model,
    anchors,
    cfg,
    tensor_chw: np.ndarray,
    device,
    conf_thr: float,
    iou_thr: float,
):
    x = (
        torch.from_numpy(tensor_chw)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        preds = model(x)

    dec = decode_predictions(
        preds,
        anchors,
        cfg["model"]["strides"],
        conf_thr=conf_thr
    )[0]

    if dec["boxes"].shape[0] == 0:
        return {
            "boxes": np.zeros(
                (0, 4),
                dtype=np.float32
            ),
            "scores": np.zeros(
                (0,),
                dtype=np.float32
            ),
            "mal_probs": np.zeros(
                (0,),
                dtype=np.float32
            ),
        }

    keep = nms(
        dec["boxes"],
        dec["scores"],
        iou_thr=iou_thr
    )

    return {
        "boxes": dec["boxes"][keep].cpu().numpy(),
        "scores": dec["scores"][keep].cpu().numpy(),
        "mal_probs": dec["mal_probs"][keep].cpu().numpy(),
    }


# --------------------------- Main -----------------------------------------


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--weights",
        required=True
    )

    ap.add_argument(
        "--input",
        required=True,
        help=".npy | .dcm folder | .nii/.nii.gz"
    )

    # NEW:
    # Optional ground-truth mask for comparison.
    ap.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional ground-truth mask .npy"
    )

    ap.add_argument(
        "--malignancy",
        type=int,
        default=None,
        help="Ground truth malignancy (0 or 1)"
    )

    ap.add_argument(
        "--out",
        default="detections_out"
    )

    ap.add_argument(
        "--conf",
        type=float,
        default=None
    )

    ap.add_argument(
        "--iou",
        type=float,
        default=None
    )

    ap.add_argument(
        "--stride_z",
        type=int,
        default=1,
        help="run every Nth slice in mode B"
    )

    ap.add_argument(
        "--patient_id",
        type=str,
        default=None
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, anchors, cfg = _load_model(
        args.weights,
        device
    )

    size = cfg["data"]["image_size"]

    conf_thr = (
        args.conf
        if args.conf is not None
        else cfg["infer"]["conf_threshold"]
    )

    iou_thr = (
        args.iou
        if args.iou is not None
        else cfg["infer"]["iou_threshold"]
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    out_dir = Path(args.out)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = out_dir / "detections.csv"

    csv_f = open(
        csv_path,
        "a",
        newline=""
    )

    writer = csv.writer(csv_f)

    if not csv_path.exists():
        writer.writerow([
        "patient_id",
        "slice",
        "x1",
        "y1",
        "x2",
        "y2",
        "conf",
        "malignancy_prob"
    ])

    inp = Path(args.input)

    # ==================================================================
    # MODE A : .npy
    # ==================================================================

    if inp.suffix == ".npy":

        arr = np.load(inp)

        # --------------------------------------------------------------
        # Image -> (3,H,W)
        # --------------------------------------------------------------

        from dataloader import _to_chw_3

        tensor = _to_chw_3(arr)

        if tensor.max() > 1.5:
            tensor = tensor / tensor.max()

        tensor = np.clip(
            tensor,
            0,
            1
        )

        if (
            tensor.shape[1] != size
            or tensor.shape[2] != size
        ):
            tensor = np.stack(
                [
                    _resize_2d(
                        c,
                        size
                    )
                    for c in tensor
                ],
                axis=0
            )

        tensor = tensor.astype(
            np.float32
        )

        # --------------------------------------------------------------
        # Detection
        # --------------------------------------------------------------

        dets = _detect_slice(
            model,
            anchors,
            cfg,
            tensor,
            device,
            conf_thr,
            iou_thr
        )

        pid = (
            args.patient_id
            or inp.stem
        )

        # --------------------------------------------------------------
        # Prediction image
        # --------------------------------------------------------------

        vis = (
            np.transpose(
                tensor,
                (1, 2, 0)
            ) * 255
        ).astype(np.uint8)

        prediction_image = draw_detections(
            vis.copy(),
            dets["boxes"],
            dets["scores"],
            dets["mal_probs"]
        )

        prediction_image = _add_title(
            prediction_image,
            "MODEL PREDICTION"
        )

        # --------------------------------------------------------------
        # Ground truth image
        # --------------------------------------------------------------

        if args.mask is not None:

            mask2d = _prepare_ground_truth_mask(
                args.mask,
                size
            )

            gt_image = _draw_ground_truth(
                vis.copy(),
                mask2d
            )

            title = "GROUND TRUTH"

            if args.malignancy is not None:
                if args.malignancy == 1:
                    title += " - MALIGNANT"
                else:
                    title += " - BENIGN"

            gt_image = _add_title(
                gt_image,
                title
            )

            # ----------------------------------------------------------
            # Side-by-side
            # ----------------------------------------------------------

            comparison = np.hstack(
                [
                    prediction_image,
                    gt_image
                ]
            )

            comparison_path = (
                out_dir
                / f"{pid}_comparison.png"
            )

            cv2.imwrite(
                str(comparison_path),
                comparison
            )

            print(
                f"Comparison image written to: "
                f"{comparison_path}"
            )

        else:

            prediction_path = (
                out_dir
                / f"{pid}.png"
            )

            cv2.imwrite(
                str(prediction_path),
                prediction_image
            )

            print(
                f"Prediction image written to: "
                f"{prediction_path}"
            )

        # --------------------------------------------------------------
        # CSV
        # --------------------------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2
        ), sc, mp in zip(
            dets["boxes"],
            dets["scores"],
            dets["mal_probs"]
        ):

            writer.writerow(
                [
                    pid,
                    0,
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    f"{sc:.4f}",
                    f"{mp:.4f}"
                ]
            )

        csv_f.close()

        print(
            f"Detections CSV written to: "
            f"{csv_path}"
        )

        return

    # ==================================================================
    # MODE B : DICOM / NIFTI
    # ==================================================================

    if inp.is_dir():

        vol, pid = _load_dicom_series(
            str(inp)
        )

    elif (
        inp.name.endswith(".nii")
        or inp.name.endswith(".nii.gz")
    ):

        vol, pid = _load_nifti(
            str(inp)
        )

    else:

        raise ValueError(
            f"unsupported input {inp}"
        )

    pid = (
        args.patient_id
        or pid
    )

    # ------------------------------------------------------------------
    # Normalize volume
    # ------------------------------------------------------------------

    vol_norm = _window_normalize(
        vol
    )

    Z = vol_norm.shape[0]

    # Resize entire volume
    vol_r = np.stack(
        [
            _resize_2d(
                vol_norm[z],
                size
            )
            for z in range(Z)
        ],
        axis=0
    )

    all_dets = []

    # ------------------------------------------------------------------
    # Process slices
    # ------------------------------------------------------------------

    for z in range(
        0,
        Z,
        args.stride_z
    ):

        chw = _stack_25d(
            vol_r,
            z
        ).astype(
            np.float32
        )

        dets = _detect_slice(
            model,
            anchors,
            cfg,
            chw,
            device,
            conf_thr,
            iou_thr
        )

        if (
            isinstance(
                dets.get("boxes"),
                np.ndarray
            )
            and len(dets["boxes"])
        ):

            # ----------------------------------------------------------
            # Prediction image
            # ----------------------------------------------------------

            vis = (
                vol_r[z] * 255
            ).astype(
                np.uint8
            )

            vis = cv2.cvtColor(
                vis,
                cv2.COLOR_GRAY2BGR
            )

            prediction_image = draw_detections(
                vis.copy(),
                dets["boxes"],
                dets["scores"],
                dets["mal_probs"]
            )

            prediction_image = _add_title(
                prediction_image,
                f"MODEL PREDICTION - slice {z}"
            )

            # ----------------------------------------------------------
            # If a mask was provided
            # ----------------------------------------------------------

            if args.mask is not None:

                mask2d = _prepare_ground_truth_mask(
                    args.mask,
                    size
                )

                gt_image = _draw_ground_truth(
                    vis.copy(),
                    mask2d
                )

                gt_image = _add_title(
                    gt_image,
                    f"GROUND TRUTH - slice {z}"
                )

                comparison = np.hstack(
                    [
                        prediction_image,
                        gt_image
                    ]
                )

                output_image = comparison

            else:

                output_image = prediction_image

            # ----------------------------------------------------------
            # Save image
            # ----------------------------------------------------------

            image_path = (
                out_dir
                / f"{pid}_z{z:04d}.png"
            )

            cv2.imwrite(
                str(image_path),
                output_image
            )

            # ----------------------------------------------------------
            # CSV
            # ----------------------------------------------------------

            for (
                x1,
                y1,
                x2,
                y2
            ), sc, mp in zip(
                dets["boxes"],
                dets["scores"],
                dets["mal_probs"]
            ):

                writer.writerow(
                    [
                        pid,
                        z,
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        f"{sc:.4f}",
                        f"{mp:.4f}"
                    ]
                )

                all_dets.append(
                    (
                        z,
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        float(sc),
                        float(mp)
                    )
                )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    csv_f.close()

    print(
        f"processed {Z} slices | "
        f"{len(all_dets)} detections | "
        f"csv -> {csv_path}"
    )


if __name__ == "__main__":
    main()