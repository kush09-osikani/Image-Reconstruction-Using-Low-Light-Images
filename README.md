# Recons_Final.py — Low-light Image Reconstruction & Change Detection

This script implements a simple linear least-squares approach to reconstruct a well-lit ("regular") RGB image from several low-light observations, and then visualizes the pixel-wise differences (positive and negative changes). It was written for the AM205 homework / image reconstruction task and expects a small local dataset of images.

## Quick summary
- Reads a single well-lit image (`regular.png`) and S low-light images (`low1.png`, `low2.png`, ...).
- Fits a per-pixel linear model using least-squares so that a linear combination of low-light pixel channels + bias approximates the regular image.
- Uses the fitted coefficients to reconstruct the regular image for two different scenes ("objects" and "bears").
- Saves reconstructed images and positive/negative difference visualizations, and prints reconstruction error norms and the fitted coefficient matrix.

## File
- `Recons_Final.py` — main script (already in this folder).

## Requirements
- Python 3.8+ recommended
- numpy
- scikit-image (imported as `skimage.io`) — for reading and writing images

Install dependencies with pip if needed:
```powershell
python -m pip install --upgrade pip
python -m pip install numpy scikit-image
```

> Note: If you previously saw `ModuleNotFoundError: No module named 'skimage'`, installing `scikit-image` with pip as shown above will fix it.

## Expected input / directory layout
The script uses hard-coded `base` paths for two scenes. Each `base` directory must contain:

- `regular.png` — the target high-quality image
- `low1.png`, `low2.png`, ..., `lowS.png` — the S low-light images

Default locations used in the script (edit these to fit your machine):

```py
base = r"C:\Users\ICL512\Desktop\semester 2 courses\reinforcement learning\am205_hw1_files\am205_hw1_files\problem5\objects\1600x1200\\"
# later overwritten for bears:
base = r"...\problem5\bears\1600x1200\\"
```

Change these `base` variables to point to your dataset folder(s) or modify the script to compute paths relative to the script location.

## How it works (algorithmic overview)
1. Read the regular image `a` (shape: N x M x 3) and S low-light images `b[k]` (same shape).
2. Flatten spatial dimensions so each pixel is a row.
3. Build matrix `A` of shape (M*N, 3*S + 1): for each low-light image and each color channel (R,G,B) we put that channel's pixel values as columns; the final column is a constant bias (ones).
4. Build target `y` of shape (M*N, 3) from the regular image (R,G,B per pixel).
5. Solve `A * F = y` in least-squares sense using `numpy.linalg.lstsq`. Result `F` has shape (3*S + 1, 3) and contains linear transform coefficients per output channel.
6. Reconstruct the image as `ao = reshape(A @ F)` and compute `change = a - ao`.
7. Save visualizations for positive differences (`pos_diff_objects.png`), negative differences (`neg_diff_objects.png`), and the reconstructed image `rec_image_objects.png`. Repeat reconstruction for the second scene (bears) using the same `F`.

## What the script prints / saves
- Prints the fitted coefficient matrix `F` to the console.
- Prints numeric reconstruction error (a squared-norm based quantity) for the `objects` and `bears` scenes.
- Saves image files (in the current working directory):
  - `pos_diff_objects.png` — positive differences (scaled for visualization)
  - `neg_diff_objects.png` — negative differences (scaled for visualization)
  - `rec_image_objects.png` — reconstruction of objects
  - `rec_image_bears.png` — reconstruction of bears

## Parameters you may want to tune
- `s = 3` — number of low-light images used. Increase or decrease depending on dataset.
- `scale_factor = 10.0` — multiplies difference images to improve visibility; reduce if saturated.
- `base` — update the base paths to point to your images.

## Running the script
From the folder containing `Recons_Final.py` run:
```powershell
python Recons_Final.py
```

For quick tests, you can reduce image resolution (or use a cropped subset) to speed up the linear solve and development/debugging.

## Troubleshooting
- ModuleNotFoundError: No module named 'skimage'
  - Install scikit-image: `python -m pip install scikit-image`
- Memory / speed issues on large images
  - The script flattens full images into large matrices (size M*N rows). For 1600x1200 images this is sizable but usually fits in memory on modern machines. To reduce memory usage:
    - Work on a downsampled/cropped image
    - Process each color channel separately (more complex if you want bias/combined coefficients)
    - Use iterative solvers or chunk-based approaches
- Wrong file path / FileNotFoundError
  - Make sure the `base` string points to the folder containing the images. Use raw strings (`r"C:\path\"`) or double backslashes on Windows.

## Possible improvements / experiments
- Use a small convolutional network to learn a nonlinear mapping instead of a linear least-squares model.
- Use a more robust regression technique (e.g., Huber loss or regularized least squares) to reduce sensitivity to outliers.
- Save the coefficient matrix `F` to disk (e.g., using `numpy.save`) so you can reuse it without refitting.
- Add a CLI (argparse) so paths and parameters (`s`, `scale_factor`, `resize`) can be provided at runtime.
- Visualize results inline with matplotlib (useful for notebooks).

## About the error metric used
The script computes a per-pixel squared-norm normalized by image area:

```
norm_val = (||change||_2)^2 / (M * N)
```

This reports a per-pixel energy of reconstruction error (summed over the three channels then normalized by number of pixels).

## License
You can keep or modify this script for your coursework. No explicit license is attached — if this is for publication, consider adding an MIT or CC license header.

---
If you want, I can:
- convert this to a command-line tool (`argparse`) and add a small example dataset loader,
- save `F` and create a small evaluation routine that computes PSNR/SSIM for reconstructions,
- or add automated-downsampling to speed up development.

Tell me which of those you'd like and I can implement it.
