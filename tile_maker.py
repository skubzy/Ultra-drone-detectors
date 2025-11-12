from pathlib import Path
import os
import numpy as np
from PIL import Image

# ========= CONFIG =========
input_path  = Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\Ultra-drone-detectors\data\drone")  # file OR folder
output_root = Path(r"C:\Users\Pc\Desktop\Desktop\Stuff\Ultra-drone-detectors\data\drone")
output_root.mkdir(parents=True, exist_ok=True)

# Heuristics (tweak if needed)
trim_bg_tol = 8          # 0..255 distance from corner background considered "background"
crop_pad    = 2          # extra pixels kept inside after cropping
rm_cbar_frac = 0.10      # remove rightmost 10% to kill colorbar if present
square_size = 256        # final square tile size (px)
force_grayscale = True   # recommended for training

# ========= HELPERS =========
def list_pngs(path: Path):
    if path.is_dir():
        return sorted([p for p in path.glob("*.png")])
    elif path.is_file() and path.suffix.lower() == ".png":
        return [path]
    else:
        raise FileNotFoundError(f"No PNG(s) at: {path}")

def to_gray(img: Image.Image):
    return img.convert("L") if force_grayscale else img.convert("RGB")

def _trim_uniform_borders(arr: np.ndarray, tol=8):
    """
    Trim outer uniform margins using the top-left corner color as background.
    Works for typical matplotlib figures with white/near-white margins.
    """
    if arr.ndim == 3:
        bg = arr[0,0,:].astype(np.int16)
        diff = np.abs(arr.astype(np.int16) - bg).max(axis=2)
    else:
        bg = int(arr[0,0])
        diff = np.abs(arr.astype(np.int16) - bg)

    rows = np.where((diff > tol).any(axis=1))[0]
    cols = np.where((diff > tol).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr  # nothing to trim

    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    r0 = max(0, r0 - crop_pad); c0 = max(0, c0 - crop_pad)
    r1 = min(arr.shape[0]-1, r1 + crop_pad)
    c1 = min(arr.shape[1]-1, c1 + crop_pad)
    return arr[r0:r1+1, c0:c1+1, ...] if arr.ndim == 3 else arr[r0:r1+1, c0:c1+1]

def _remove_right_colorbar(arr: np.ndarray, frac=0.10):
    """Drop a small right strip (colorbar); keep left (1-frac)."""
    W = arr.shape[1]
    keep = int(round(W * (1.0 - frac)))
    if keep < 1: 
        keep = W
    return arr[:, :keep, ...]


# ========= MAIN =========
pngs = list_pngs(input_path)

for png in pngs:
    img = Image.open(png)
    img = to_gray(img)
    arr = np.array(img)

    # 1) Trim outer white margins (axes labels, titles, figure frame)
    arr = _trim_uniform_borders(arr, tol=trim_bg_tol)

    # 2) Remove right colorbar if present (heuristic: last ~10% of width)
    arr = _remove_right_colorbar(arr, frac=rm_cbar_frac)

    # 3) Optional: trim again (some fig styles add a thin border around axes)
    arr = _trim_uniform_borders(arr, tol=trim_bg_tol)

    H, W = arr.shape[:2]
    if W < 10:
        print(f"Too narrow after crop, skipping: {png}")
        continue

    # 4) Split into 10 non-overlapping vertical tiles (time axis)
    num_tiles = 10
    tile_w = W // num_tiles
    if tile_w < 1:
        print(f"Image too small for 10 tiles: {png}")
        continue

    out_dir = output_root / png.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(num_tiles):
        x0 = k * tile_w
        x1 = x0 + tile_w
        tile = arr[:, x0:x1]  # H x tile_w

        # 5) Resize each tile to a square for the model
        tile_img = Image.fromarray(tile)
        tile_img = tile_img.resize((square_size, square_size), resample=Image.BICUBIC)

        # 6) Save
        out_path = out_dir / f"{png.stem}_tile{k:02d}.png"
        tile_img.save(out_path)

    print(f"{png.name}: saved {num_tiles} tiles -> {out_dir}")

print("Complete.")
