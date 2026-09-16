import numpy as np
import pandas as pd

from dataloader import LIDCDetectionDataset


def test_single_slice_mode_uses_center_slice(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    patient = "p001"
    center_slice = 10
    img = np.stack(
        [
            np.full((8, 8), 0.1, dtype=np.float32),
            np.full((8, 8), 0.5, dtype=np.float32),
            np.full((8, 8), 0.9, dtype=np.float32),
        ],
        axis=0,
    )
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    np.save(images_dir / f"{patient}_{center_slice}.npy", img)
    np.save(masks_dir / f"{patient}_{center_slice}.npy", mask)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame([
        {"patient_id": patient, "center_slice": center_slice}
    ]).to_csv(csv_path, index=False)

    ds = LIDCDetectionDataset(
        csv_path,
        images_dir,
        masks_dir,
        image_size=8,
        augment=False,
        in_channels=1,
    )

    image, _ = ds[0]
    assert image.shape == (1, 8, 8)
    np.testing.assert_array_equal(image[0], img[1])
