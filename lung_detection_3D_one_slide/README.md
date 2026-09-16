# 3D-aware lung detector prototype

This directory contains a 3D-aware variant inspired by the 2D YOLO-style detector in the parent project.

The goal is to keep the same project structure and training logic, while adding volumetric context before the final 2D detection head. This is a practical intermediate step between a pure 2D detector and a full 3D detector.

## Main ideas

- The training data can still be read from the same CSV-based LIDC annotations.
- Each sample is transformed into a small 3D context volume using neighboring slices around the center slice.
- A lightweight 3D encoder extracts volumetric features.
- The depth dimension is collapsed to a central 2D feature map.
- The original YOLO-style detection head is reused for box/object/malignancy prediction.

This is not a full 3D bounding-box detector yet, but it is a valid 3D-aware prototype for a 256x256x256 CT workflow.

## Recommended next step

For a true 256x256x256 detector, replace the slice CSV workflow with a patient-level volume dataset and switch the final box head to a true 3D anchor head.

## Files

- `configs/default.yaml`: 3D-aware default configuration
- `dataloader.py`: volume-aware dataset and loader
- `model.py`: 3D encoder + 2D detection head
- `train_test.py`: training / testing entrypoint
