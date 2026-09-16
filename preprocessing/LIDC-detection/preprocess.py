import os
import numpy as np
if not hasattr(np, "int6416"):
    np.int6416 = np.int16  # Compatibility alias for pylidc on NumPy 2.x
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2  # Used for efficient 2D spatial resizing

# --- CONFIGURATION ---
# Point PyLIDC to the cluster's LIDC dataset
LIDC_DATA_PATH = "/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/LIDC_IDRI/raw_data/TCIA_LIDC-IDRI_20200921/LIDC-IDR"
os.environ['PYLIDC_DATA'] = LIDC_DATA_PATH

OUTPUT_DIR = "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def preprocess_volume(vol, min_hu=-1000, max_hu=400):
    """Clips HU values and normalizes the volume to [0, 1]."""
    vol = np.clip(vol, min_hu, max_hu)
    vol = (vol - min_hu) / (max_hu - min_hu)
    return vol.astype(np.float32)

def resize_channels(array, target_shape=(256, 256), is_mask=False):
    """Resizes a (H, W, 3) array channel-by-channel to (target_shape, 3)."""
    resized_channels = []
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    for i in range(3):
        channel_resized = cv2.resize(array[:, :, i], target_shape, interpolation=interpolation)
        resized_channels.append(channel_resized)
        
    return np.stack(resized_channels, axis=-1)

# --- PHASE 1: SPLIT PATIENTS (80/20 Grouped Split) ---
print("Fetching all scans from pylidc database...")
scans = pl.query(pl.Scan).all()

patient_ids = list(set([scan.patient_id for scan in scans]))
patient_ids.sort()

train_patients, test_patients = train_test_split(patient_ids, test_size=0.20, random_state=42)
train_patients_set = set(train_patients)

print(f"Total Patients: {len(patient_ids)} | Train: {len(train_patients)} | Test: {len(test_patients)}")

# --- PHASE 2: PROCESSING & GROUPING NODULES BY CENTER SLICE ---
metadata_records = []

print("Extracting slices, grouping nodules, and formatting to (1, 1, 256, 256, 3)...")
for scan in tqdm(scans):
    pid = scan.patient_id
    split_group = "train" if pid in train_patients_set else "test"
    
    # Load and preprocess full 3D scan volume
    vol = scan.to_volume()
    preprocessed_vol = preprocess_volume(vol)
    max_slices = preprocessed_vol.shape[2]
    
    nodules = scan.cluster_annotations()
    if not nodules:
        continue
        
    # Group nodule consensus masks by their calculated target center slice
    slice_groups = {}
    
    for nodule_idx, nodule_annotations in enumerate(nodules):
        cmask, cbbox, masks = consensus(nodule_annotations, clevel=0.5)
        
        # Extract malignancy scores from annotations (scale 1-5)
        malignancies = [ann.malignancy for ann in nodule_annotations if ann.malignancy is not None]
        avg_malignancy = np.mean(malignancies) if malignancies else 0
        is_malignant = 1 if avg_malignancy >= 3 else 0  # Malignant if avg >= 3
        
        z_start, z_stop = cbbox[2].start, cbbox[2].stop
        center_slice = int((z_start + z_stop) / 2)
        
        if center_slice not in slice_groups:
            slice_groups[center_slice] = []
            
        # Store information needed to draw this mask onto the shared slice stack later
        slice_groups[center_slice].append({
            'nodule_idx': nodule_idx,
            'cmask': cmask,
            'cbbox': cbbox,
            'z_start': z_start,
            'z_stop': z_stop,
            'malignancy_score': avg_malignancy,
            'is_malignant': is_malignant
        })

    # Generate files for each unique center slice context found in this patient
    for center_slice, associated_nodules in slice_groups.items():
        
        # 1. Grab the 3-slice configuration indices
        slice_indices = [
            np.clip(center_slice - 1, 0, max_slices - 1),
            center_slice,
            np.clip(center_slice + 1, 0, max_slices - 1)
        ]
        
        # 2. Extract 3-channel slice array from CT volume
        slice_stack = preprocessed_vol[:, :, slice_indices] # Shape: (512, 512, 3)
        
        # 3. Create a blank matching 3-channel mask canvas
        mask_stack = np.zeros_like(slice_stack, dtype=np.uint8) # Shape: (512, 512, 3)
        
        # Map ALL overlapping nodules belonging to this slice setup
        nodule_indices_present = []
        malignancy_scores = []
        is_malignant_list = []
        for nodule in associated_nodules:
            nodule_indices_present.append(nodule['nodule_idx'])
            malignancy_scores.append(nodule['malignancy_score'])
            is_malignant_list.append(nodule['is_malignant'])
            cbbox = nodule['cbbox']
            z_start = nodule['z_start']
            z_stop = nodule['z_stop']
            cmask = nodule['cmask']
            
            for idx, s_idx in enumerate(slice_indices):
                if z_start <= s_idx < z_stop:
                    local_z = s_idx - z_start
                    # Use np.maximum to seamlessly stack overlapping masks
                    mask_stack[cbbox[0], cbbox[1], idx] = np.maximum(
                        mask_stack[cbbox[0], cbbox[1], idx], 
                        cmask[:, :, local_z].astype(np.uint8)
                    )

        # 4. Downsample dimensions from (512, 512, 3) to (256, 256, 3)
        slice_stack_rescaled = resize_channels(slice_stack, target_shape=(256, 256), is_mask=False)
        mask_stack_rescaled = resize_channels(mask_stack, target_shape=(256, 256), is_mask=True)
        
        # 5. Expand tensor shapes to match your deep learning requirement: (B, 1, H, W, C)
        # Here, B=1 for saving individual files
        final_ct_tensor = np.expand_dims(slice_stack_rescaled, axis=(0, 1)) # Shape: (1, 1, 256, 256, 3)
        final_mask_tensor = np.expand_dims(mask_stack_rescaled, axis=(0, 1)) # Shape: (1, 1, 256, 256, 3)
        
        # Save setup files
        image_filename = f"{pid}_slice_{center_slice}_img.npy"
        mask_filename = f"{pid}_slice_{center_slice}_mask.npy"
        
        image_save_path = os.path.join(IMAGES_DIR, image_filename)
        mask_save_path = os.path.join(MASKS_DIR, mask_filename)
        
        np.save(image_save_path, final_ct_tensor)
        np.save(mask_save_path, final_mask_tensor)
        
        # Log entry
        # Determine overall malignancy: 1 if ANY nodule is malignant, 0 if all benign
        group_malignancy = 1 if any(is_malignant_list) else 0
        avg_malignancy_score = np.mean(malignancy_scores) if malignancy_scores else 0
        
        metadata_records.append({
            "patient_id": pid,
            "center_slice": center_slice,
            "grouped_nodule_idxs": str(nodule_indices_present),
            "num_grouped_nodules": len(nodule_indices_present),
            "avg_malignancy_score": round(avg_malignancy_score, 3),
            "malignancy": group_malignancy,  # 1 if malignant, 0 if benign
            "image_path": image_save_path,
            "mask_path": mask_save_path,
            "image_shape": str(final_ct_tensor.shape),
            "mask_shape": str(final_mask_tensor.shape),
            "split": split_group
        })

# --- PHASE 3: CREATE TRAIN & TEST CSV FILES ---
df = pd.DataFrame(metadata_records)

df_train = df[df["split"] == "train"].drop(columns=["split"])
df_test = df[df["split"] == "test"].drop(columns=["split"])

df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("\nPreprocessing Complete!")
print(f"Saved {len(df_train)} grouped train sample files -> {OUTPUT_DIR}/train.csv")
print(f"Saved {len(df_test)} grouped test sample files -> {OUTPUT_DIR}/test.csv")