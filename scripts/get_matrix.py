#!/usr/bin/env python3

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
#from skimage.filters import gaussian
import pandas as pd
from scipy.ndimage import (
  gaussian_filter,
  zoom,
  distance_transform_edt)
import skimage.exposure as (skie, rescale_intensity)
from sklearn.linear_model import HuberRegressor
from skimage.transform import resize
import gc

# =========================
# USER INPUTS
# =========================

MARKER_FILES = {
    "PGP95": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_1.0.4_R000_Cy3_PGP9-5-AF555_FINAL_AFR_F.ome.tif",
    "CD45": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_1.0.4_R000_Cy5_CD45-AF647_FINAL_AFR_F.ome.tif",
    "CD10": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_1.0.4_R000_Cy7_CD10-CF750_FINAL_AFR_F.ome.tif",
    "DAPI_R1": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_1.0.4_R000_DAPI__FINAL_F.ome.tif",
    "KRT8": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_2.0.4_R000_Cy5_KRT8-18-AF647_FINAL_AFR_F.ome.tif",
    "DAPI_R2": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_2.0.4_R000_DAPI__FINAL_F.ome.tif",
    "CD20": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_2.0.4_R000_FITC_CD20-AF488_FINAL_AFR_F.ome.tif",
}
AUTOFLUORESCENCE_FILE = "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260090-CD26039_S19-06413-5580-US-4/raw/CD26039_1.0.1_R000_DAPI_AF_F.ome.tif"

DAPI_MARKERS = ["DAPI_R1", "DAPI_R2"]


BIN_SIZE = 20   # at 50 KRT8 fragments

OUTPUT_DIR = "/media/Lawrenson_Lab_NAS/uthscsa/group_data/CosMx_temp/SL260090/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================
# 1. FUNCTIONS
# =========================
def load_tif(path):

    img = tifffile.imread(path)

    if img.ndim > 2:
        img = img.squeeze()

    return img.astype(np.float32)


def tile_normalize_fast(
    img,
    tile_size=512,
    smooth_sigma=500,
    eps=1e-6
):
    H, W = img.shape
    # -------------------------------------------------------
    # Number of tiles
    # -------------------------------------------------------
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size

    tile_map = np.zeros(
        (n_tiles_y, n_tiles_x),
        dtype=np.float32)
    # -------------------------------------------------------
    # Fast per-tile statistics
    # -------------------------------------------------------
    for ty in range(n_tiles_y):
        y0 = ty * tile_size
        y1 = min((ty + 1) * tile_size, H)

        for tx in range(n_tiles_x):
            x0 = tx * tile_size
            x1 = min((tx + 1) * tile_size, W)
            tile = img[y0:y1, x0:x1]
            # Robust illumination estimate
            tile_map[ty, tx] = np.percentile(tile, 50)

    # -------------------------------------------------------
    # Smooth low-resolution map
    # -------------------------------------------------------
    tile_map = gaussian_filter(
        tile_map,
        sigma=smooth_sigma)
    # -------------------------------------------------------
    # Normalize around global median
    # -------------------------------------------------------
    tile_map /= np.median(tile_map)
    # -------------------------------------------------------
    # Upsample illumination map
    # -------------------------------------------------------
    background = resize(
        tile_map,
        (H, W),
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)
    # -------------------------------------------------------
    # Correct image
    # -------------------------------------------------------
    corrected = img / (background + eps)
    corrected = rescale_intensity(corrected,
                                  in_range=(np.percentile(corrected, 15),
                                            np.percentile(corrected, 100)),
                                  out_range=(0,1))

    return corrected


def illumination_correct_fast(img,sigma=80,downscale=8):
    small = img[::downscale, ::downscale]
    background_small = gaussian_filter(small,
                                       sigma=sigma / downscale)
    background = zoom(background_small,downscale,order=1)
    background = background[:img.shape[0],:img.shape[1]]
    corrected = img / (background + 1e-6)
    corrected= np.clip(1-corrected, 0,1)
    corrected = rescale_intensity(corrected,
                                  in_range=(np.percentile(corrected, 15),
                                            np.percentile(corrected, 100)),
                                  out_range=(0,1))
    return corrected


def bin_image(img, bin_size):
    H, W = img.shape
    H_trim = (H // bin_size) 
    W_trim = (W // bin_size)
    img = img[:H_trim* bin_size, :W_trim* bin_size]

    binned = img.reshape(
        H_trim,
        bin_size,
        W_trim,
        bin_size).mean(axis=(1,3))
    return binned


def compute_alpha(marker_vec, af_vec, dapi_vec):
    bg_mask = (
        (marker_vec < np.percentile(marker_vec, 30)) &
        (dapi_vec < np.percentile(dapi_vec, 50))
    )
    X = af_vec[bg_mask].reshape(-1,1)
    y = marker_vec[bg_mask]
    model = HuberRegressor()
    model.fit(X, y)
    alpha = model.coef_[0]
    # conservative cap
    alpha = np.clip(alpha, 0, 0.8)
    return alpha

# =========================
# 2. DAPI                                            #
# =========================
print("Reading DAPI files...")
dapi_r1 = load_tif(MARKER_FILES["DAPI_R1"])
dapi_r2 = load_tif(MARKER_FILES["DAPI_R2"])
print("Correcting illumination...")
dapi_r1=illumination_correct_fast(dapi_r1)
dapi_r2=illumination_correct_fast(dapi_r2)
print("...done")

dapi_avg = (dapi_r1 + dapi_r2) / 2
#plt.figure(figsize=(8,8))
#plt.imshow(dapi_avg, cmap='inferno',
#           vmin=np.percentile(dapi_avg, 5),
#           vmax=np.percentile(dapi_avg, 99))
#plt.colorbar()
#plt.title("DAPI")
#plt.savefig(os.path.join(OUTPUT_DIR, "DAPI.png"))    

# =========================
# 3. AF
# =========================
print("Reading AF file")
af_img = load_tif(AUTOFLUORESCENCE_FILE)
af_img=illumination_correct_fast(af_img,80)
#plt.figure(figsize=(8,8))
#plt.imshow(af_img, cmap='inferno',
#           vmin=np.percentile(af_img, 5),
#           vmax=np.percentile(af_img, 99))
#plt.colorbar()
#plt.title("AF")
#plt.savefig(os.path.join(OUTPUT_DIR, "AF.png"))    

# =========================
# 4. MASK
# =========================
print("Building tissue mask...")
print("AF threshold:", np.percentile(af_img, 75))
print("DAPI threshold:", np.percentile(dapi_avg, 75))
pixel_mask = (
    (af_img > np.percentile(af_img, 75)) |
    (dapi_avg > np.percentile(dapi_avg, 75))
)
#plt.figure(figsize=(8,8))
#plt.imshow(pixel_mask, cmap='inferno')
#plt.colorbar()
#plt.title("Mask")

distance_map = distance_transform_edt(pixel_mask)
print("Distance threshold:", np.percentile(distance_map, 95))
plt.figure(figsize=(8,8))
plt.imshow(distance_map, cmap='inferno',
           vmin=np.percentile(distance_map, 5),
           vmax=np.percentile(distance_map, 99))
plt.colorbar()
plt.title("Distance")
plt.savefig(os.path.join(OUTPUT_DIR, "Distance.png"))

pixel_mask = (distance_map >= 40)

print("Computing occupancy...")
occupancy = bin_image(pixel_mask,BIN_SIZE)
meta_mask = occupancy > .7

neg_mask =np.clip(1-meta_mask, 0,None)
plt.figure(figsize=(8,8))
plt.imshow(neg_mask, cmap='inferno')
plt.colorbar()
plt.title("Mask")
plt.savefig(os.path.join(OUTPUT_DIR, "Neg_mask.png"))

# bin mask + get coordinates
print("Binning AF and DAPI...")
af_bin = bin_image(af_img,BIN_SIZE)
dapi_bin = bin_image(dapi_avg,BIN_SIZE)

mask_flat = meta_mask.reshape(-1)
af_flat = af_bin.reshape(-1)[mask_flat]
dapi_flat = dapi_bin.reshape(-1)[mask_flat]
# ============================================================
# 5. MARKERS
# ============================================================
marker_names = [m for m in MARKER_FILES if m not in DAPI_MARKERS]

print("Processing markers...")

results = []

coords_y, coords_x = np.where(meta_mask)

for marker in marker_names:
    print(f"Processing {marker}")
    img = load_tif(MARKER_FILES[marker])
    img = tile_normalize_fast(img)
    img_bin = bin_image(img,BIN_SIZE)
    #img_neg=np.clip(img_bin-neg_mask,0,None)
    #marker_vec = img_neg.reshape(-1)[mask_flat]
    
    marker_vec = img_bin.reshape(-1)[mask_flat]

    alpha = compute_alpha(
        marker_vec,
        af_flat,
        dapi_flat
    )
    print(f"alpha = {alpha:.3f}")

    corrected = marker_vec - alpha * af_flat
#    img_corr=np.clip(img_neg-alpha*af_bin,0,None)
    img_corr=np.clip(img_bin-alpha*af_bin,0,None)
    plt.tight_layout()
    plt.figure(figsize=(8,8))
    plt.imshow(img_corr, cmap='inferno',
               vmin=np.percentile(img_corr, 5),
               vmax=np.percentile(img_corr, 99))
    plt.colorbar()
    plt.title(marker)

    corrected = np.clip(corrected,0,None)
    results.append(corrected)

    gc.collect()
# ============================================================
# 6. SAVE FINAL MATRIX
# ============================================================

# ============================================================
# SAVE FINAL MATRIX
# ============================================================

#final_df = pd.concat(
#    results,
#    ignore_index=True
#)
pixel_matrix = np.stack(results, axis=1)

df = pd.DataFrame(pixel_matrix, columns=marker_names)
df["x"] = coords_x
df["y"] = coords_y
df["AF"]=af_flat
df = df[["x", "y","AF"] + marker_names]

out_csv = os.path.join(
    OUTPUT_DIR,
    "meta_pixel_matrix.csv"
)

df.to_csv(
    out_csv,
    index=False
)

print("Done.")
print(df.shape)
print(f"Saved: {out_csv}")
