#!/usr/bin/env python3

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
#from skimage.filters import gaussian
import pandas as pd
from scipy.ndimage import zoom
from sklearn.linear_model import HuberRegressor
from skimage.filters import threshold_otsu

import gc

# =========================
# USER INPUTS
# =========================

MARKER_FILES = {
    "PGP95": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy3_PGP9-5-AF555_FINAL_AFR_F.ome.tif",
    "CD45": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy5_CD45-AF647_FINAL_AFR_F.ome.tif",
    "CD10": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_Cy7_CD10-CF750_FINAL_AFR_F.ome.tif",
    "DAPI_R1": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.4_R000_DAPI__FINAL_F.ome.tif",
    "KRT8": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/O13-US-4/raw/CD26037_2.0.4_R000_Cy5_KRT8-18-AF647_FINAL_AFR_F.ome.tif",
    "DAPI_R2": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_2.0.4_R000_DAPI__FINAL_F.ome.tif",
    "CD20": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_2.0.4_R000_FITC_CD20-AF488_FINAL_AFR_F.ome.tif",
}
AUTOFLUORESCENCE_FILE = "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260088-CD26037_S-19-56318-B1-BEME-O13-US-4/raw/CD26037_1.0.1_R000_DAPI_AF_F.ome.tif"

DAPI_MARKERS = ["DAPI_R1", "DAPI_R2"]


BIN_SIZE = 40   # at 50 KRT8 fragments

OUTPUT_DIR = "/media/Lawrenson_Lab_NAS/uthscsa/group_data/CosMx_temp/SL260088/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================
# 1. FUNCTIONS
# =========================
def load_tif(path):

    img = tifffile.imread(path)

    if img.ndim > 2:
        img = img.squeeze()

    return img.astype(np.float32)


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


def compute_alpha(marker_vec, af_vec):#SLOW
    bg_mask = (marker_vec < np.percentile(marker_vec, 50))
    X = af_vec[bg_mask].reshape(-1,1)
    y = marker_vec[bg_mask]
    model = HuberRegressor()
    model.fit(X, y)
    alpha = model.coef_[0]
    # conservative cap
    alpha = np.clip(alpha, 0.01, 0.8)
    return alpha

# FAST
#def compute_alpha(marker_vec, af_vec):
#    m = np.percentile(marker_vec, 90)
#    a = np.percentile(af_vec, 90)
#    if a == 0:
#        return 0
#    return (m / a) * 0.7
# =========================
# 2. AF
# =========================
print("Reading AF file")
af_img = load_tif(AUTOFLUORESCENCE_FILE)
af_img=np.arcsinh(af_img/ 5)
#small = af_img[::8, ::8]
#plt.figure(figsize=(8,8))
#plt.imshow(af_img, cmap='inferno',
#           vmin=np.percentile(af_img, 5),
#           vmax=np.percentile(af_img, 99))
#plt.colorbar()
#plt.title("AF")

#plt.savefig(os.path.join(OUTPUT_DIR, "AF.png"))    

# =========================
# 3. MASK
# =========================
print("Building tissue mask...")
thr_af=np.percentile(af_img, 75)
#thr_dapi=np.percentile(dapi_avg, 75)
#print("AF threshold:", thr_af)
#print("DAPI threshold:", thr_dapi)
pixel_mask = (af_img > thr_af) 
#small = pixel_mask[::8, ::8]
#plt.figure(figsize=(8,8))
#plt.imshow(pixel_mask, cmap='inferno')
#plt.colorbar()
#plt.title("Mask")

print("Computing occupancy...")
occupancy = bin_image(pixel_mask,BIN_SIZE)
meta_mask = occupancy > .8# with dapi mask was .4 

neg_mask =np.clip(1-meta_mask, 0,None)
small = neg_mask[::8, ::8]
plt.figure(figsize=(8,8))
plt.imshow(neg_mask, cmap='inferno')
plt.colorbar()
plt.title("Mask")
plt.savefig(os.path.join(OUTPUT_DIR, "Neg_mask.png"))

# bin mask + get coordinates
print("Binning AF and DAPI...")
af_bin = bin_image(af_img,BIN_SIZE)
#dapi_bin = bin_image(dapi_avg,BIN_SIZE)

mask_flat = meta_mask.reshape(-1)
af_flat = af_bin.reshape(-1)[mask_flat]
#dapi_flat = dapi_bin.reshape(-1)[mask_flat]

# ============================================================
# 5. MARKERS
# ============================================================
marker_names = [m for m in MARKER_FILES]

print("Processing markers...")

results = []

coords_y, coords_x = np.where(meta_mask)

for marker in marker_names:
    print(f"Processing {marker}")
    img = load_tif(MARKER_FILES[marker])
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 8))
    im1=ax1.imshow(img, cmap='inferno',
               vmin=np.percentile(img, 5),
               vmax=np.percentile(img, 99))
        
    img = np.arcsinh(img/ 5)
    img_bin = bin_image(img,BIN_SIZE)
    marker_vec = img_bin.reshape(-1)[mask_flat]

    alpha = compute_alpha(
        marker_vec,
        af_flat
    )
    print(f"alpha = {alpha:.3f}")
    img_corr=np.clip(img_bin-alpha*af_bin,0,None)

    #thresh = threshold_otsu(img_corr)
    binary_image = img_corr < np.percentile(img_corr,0.9)
    img=img_corr.copy()
    img[binary_image==1] = 0

    im2=ax2.imshow(img, cmap='inferno',
               vmin=np.percentile(img, 5),
               vmax=np.percentile(img, 99))
    plt.tight_layout() # Adjusts spacing to prevent overlap
    plt.show()
    
    marker_vec = img_corr.reshape(-1)[mask_flat]
    results.append(marker_vec)

    gc.collect()
# ============================================================
# 5. SAVE FINAL MATRIX
# ============================================================
pixel_matrix = np.stack(results, axis=1)

df = pd.DataFrame(pixel_matrix, columns=marker_names)
df["x"] = coords_x
df["y"] = coords_y
#df["AF"]=af_flat
df = df[["x", "y"] + marker_names]

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
