import os
import numpy as np
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
IMAGE_PATH = "/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/LIDC-detection/images/LIDC-IDRI-0001_slice_90_img.npy"
# Deriving the matching mask path automatically based on your naming convention
MASK_PATH = "/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/LIDC-detection/masks/LIDC-IDRI-0001_slice_90_mask.npy"

def load_and_squeeze(file_path):
    """Loads the tensor and removes the batch/channel dimensions (1, 1, 256, 256, 3) -> (256, 256, 3)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")
    
    tensor = np.load(file_path)
    # Squeeze out the dimensions of size 1 at axis 0 and 1
    squeezed = np.squeeze(tensor, axis=(0, 1))
    return squeezed

# --- LOAD DATA ---
try:
    img_stack = load_and_squeeze(IMAGE_PATH)   # Shape: (256, 256, 3)
    mask_stack = load_and_squeeze(MASK_PATH) # Shape: (256, 256, 3)
    print(f"Loaded Image Stack Shape: {img_stack.shape}")
    print(f"Loaded Mask Stack Shape: {mask_stack.shape}")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- PLOTTING ---
# We create a 3x3 grid: 
# Row 1: Preprocessed CT slices
# Row 2: Binary Consensus Masks
# Row 3: Transparent Overlay (CT + Mask)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
slice_names = ['Slice: Center - 1', 'Slice: Center', 'Slice: Center + 1']

for i in range(3):
    # Extract the individual 2D channel slice
    ct_slice = img_stack[:, :, i]
    mask_slice = mask_stack[:, :, i]
    
    # 1. Plot Raw CT Image Channel
    axes[0, i].imshow(ct_slice, cmap='gray')
    axes[0, i].set_title(f"{slice_names[i]}\n(CT Scan)")
    axes[0, i].axis('off')
    
    # 2. Plot Binary Mask Channel
    axes[1, i].imshow(mask_slice, cmap='bone')
    axes[1, i].set_title(f"{slice_names[i]}\n(Nodule Mask)")
    axes[1, i].axis('off')
    
    # 3. Plot Alpha Blended Overlay
    # Render base CT image
    axes[2, i].imshow(ct_slice, cmap='gray')
    
    # Construct a custom red mask for visual overlay
    # Where mask is 1, make it red [1, 0, 0], where 0 keep it transparent
    red_mask = np.zeros((*mask_slice.shape, 4)) # RGBA
    red_mask[mask_slice == 1] = [1, 0, 0, 0.4]  # Red color with 40% opacity
    
    axes[2, i].imshow(red_mask)
    axes[2, i].set_title(f"{slice_names[i]}\n(Overlay)")
    axes[2, i].axis('off')

plt.suptitle(f"Visualization for Patient LIDC-IDRI-0069 (Slice 40 context)", fontsize=16, weight='bold')
plt.tight_layout()

# Save the visualization figure in the output directory
output_plot_path = "/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/LIDC-detection/"
plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
print(f"Verification plot cleanly saved to: {output_plot_path}")

# Display the plot
plt.show()