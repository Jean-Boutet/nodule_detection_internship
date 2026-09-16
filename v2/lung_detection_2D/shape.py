import numpy as np

arr = np.load("/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/images/LIDC-IDRI-0001_slice_90_img.npy")

print(arr.shape)
print(arr.dtype)