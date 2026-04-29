#!/usr/bin/env python3

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import pandas as pd

# =========================
# USER INPUTS
# =========================

MARKER_FILES = {
    "PGP95": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy3_PGP9-5-AF555_FINAL_AFR_F.ome.tif",
    "CD45": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy5_CD45-AF647_FINAL_AFR_F.ome.tif",
    "CD10": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy7_CD10-CF750_FINAL_AFR_F.ome.tif",
    "DAPI_R1": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_DAPI__FINAL_F.ome.tif",
    "KRT8": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_2.0.4_R000_Cy5_KRT8-18-AF647_FINAL_AFR_F.ome.tif",
    "DAPI_R2": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_2.0.4_R000_DAPI__FINAL_F.ome.tif",
    "CD20": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_2.0.4_R000_FITC_CD20-AF488_FINAL_AFR_F.ome.tif",
}
AUTOFLUORESCENCE_FILE = "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.1_R000_DAPI_AF_F.ome.tif"

DAPI_MARKERS = ["DAPI_R1", "DAPI_R2"]

ARCSINH_COFACTOR = 5
GAUSSIAN_SIGMA = 1
GAUSSIAN_SIGMA_AF = 3
BIN_SIZE = 40   # at 50 KRT8 fragments

AF_SCALING_FACTOR = 0.6

OUTPUT_DIR = "/media/Lawrenson_Lab_NAS/uthscsa/group_data/CosMx_temp/SL260088/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FUNCTIONS
# =========================

def bin_image(image, bin_size):
    H, W = image.shape
    H2 = H // bin_size
    W2 = W // bin_size
    image = image[:H2*bin_size, :W2*bin_size]
    return image.reshape(H2, bin_size, W2, bin_size).mean(axis=(1,3))

def bin_image_with_coords(image, bin_size):
    binned = bin_image(image, bin_size)
    H2, W2 = binned.shape

    y_coords = (np.arange(H2) * bin_size + bin_size // 2)
    x_coords = (np.arange(W2) * bin_size + bin_size // 2)

    xv, yv = np.meshgrid(x_coords, y_coords)
    return binned, xv, yv

def compute_alpha(marker_vec, af_vec):
    m = np.percentile(marker_vec, 90)
    a = np.percentile(af_vec, 90)
    if a == 0:
        return 0
    return (m / a) * AF_SCALING_FACTOR
  
  
# =========================
# 2. DAPI MASK                                               #
# =========================

dapi_r1 = tifffile.imread(MARKER_FILES["DAPI_R1"]).astype(np.float32)
dapi_r2 = tifffile.imread(MARKER_FILES["DAPI_R2"]).astype(np.float32)
af_img = tifffile.imread(AUTOFLUORESCENCE_FILE).astype(np.float32)

dapi_r1 = np.arcsinh(dapi_r1 / ARCSINH_COFACTOR)
dapi_r2 = np.arcsinh(dapi_r2 / ARCSINH_COFACTOR)

# QC plot: difference
plt.imshow(dapi_r1 - dapi_r2, cmap='bwr')
plt.colorbar()
plt.title("DAPI Difference (R1 - R2)")
plt.savefig(os.path.join(OUTPUT_DIR, "dapi_difference.png"))
plt.close()

dapi_avg = (dapi_r1 + dapi_r2) / 2
#diff = dapi_r1 - dapi_r2
dapi_avg = gaussian(dapi_avg, sigma=GAUSSIAN_SIGMA)
plt.imshow(dapi_avg, cmap='inferno',
           vmin=np.percentile(dapi_avg, 5),
           vmax=np.percentile(dapi_avg, 99))
plt.colorbar()
plt.title("DAPI")
plt.savefig(os.path.join(OUTPUT_DIR, "DAPI.png"))
plt.close()

threshold = np.percentile(dapi_avg, 60)
print("DAPI threshold:", threshold)

dapi_mask = dapi_avg > threshold
print(f"Mask retains {dapi_mask.mean()*100:.2f}% of pixels")

# =========================
# 3. BUILD MASK (AF-based)
# =========================

af_img = tifffile.imread(AUTOFLUORESCENCE_FILE).astype(np.float32)
af_img = np.arcsinh(af_img / ARCSINH_COFACTOR)
af_img = gaussian(af_img, sigma=GAUSSIAN_SIGMA_AF)

# QC AF plot
plt.imshow(af_img, cmap='inferno',
           vmin=np.percentile(af_img, 5),
           vmax=np.percentile(af_img, 99))
plt.colorbar()
plt.title("Autofluorescence")
plt.savefig(os.path.join(OUTPUT_DIR, "autofluorescence.png"))
plt.close()


print("Building tissue mask...")

threshold = np.percentile(af_img, 50)
print("AF threshold:", threshold)

af_mask = af_img > threshold
print(f"Mask retains {af_mask.mean()*100:.2f}% of superpixels")

mask = af_mask | dapi_mask
# bin mask + get coordinates
mask_binned, xv, yv = bin_image_with_coords(mask.astype(float), BIN_SIZE)
mask = mask_binned > 0.5

mask_flat = mask.reshape(-1)
x_flat = xv.reshape(-1)[mask_flat]
y_flat = yv.reshape(-1)[mask_flat]

# bin AF
af_img = bin_image(af_img, BIN_SIZE)
af_flat = af_img.reshape(-1)[mask_flat]

dapi_binned = bin_image(dapi_avg, BIN_SIZE)
dapi_flat=dapi_binned.reshape(-1)[mask_flat]

# =========================
# 4. PROCESS MARKERS
# =========================

marker_names = [m for m in MARKER_FILES if m not in DAPI_MARKERS]
pixel_data = []

print("\nProcessing markers with AF subtraction...")

for marker in marker_names:
    print(f"Processing {marker}...")

    img = tifffile.imread(MARKER_FILES[marker]).astype(np.float32)
    img = np.arcsinh(img / ARCSINH_COFACTOR)
    img = gaussian(img, sigma=GAUSSIAN_SIGMA)

    img = bin_image(img, BIN_SIZE)
    marker_vec = img.reshape(-1)[mask_flat]

    # compute AF scaling
    alpha = compute_alpha(marker_vec, af_flat)
    print(f"  alpha = {alpha:.3f}")

    corrected = marker_vec - alpha * af_flat
    corrected = np.clip(corrected, 0, None)

    pixel_data.append(corrected)

# =========================
# 5. BUILD FINAL MATRIX
# =========================

print("\nBuilding matrix...")

pixel_matrix = np.stack(pixel_data, axis=1)

df = pd.DataFrame(pixel_matrix, columns=marker_names)
df["x"] = x_flat
df["y"] = y_flat
df["DAPI_avg"]=dapi_flat

df = df[["x", "y","DAPI_avg"] + marker_names]

df.to_csv(os.path.join(OUTPUT_DIR, "pixel_matrix.csv"), index=False)

print(f"Matrix shape: {df.shape}")


