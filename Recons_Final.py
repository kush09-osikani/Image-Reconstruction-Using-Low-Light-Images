import numpy as np
from skimage import io

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

# Number of low-light images to use (three lighting conditions)
s = 3

# Image resolution used for this problem
sz = "1600x1200"

# ----------------------------------------------------------------
# STEP 1: Read in the "objects" images
# ----------------------------------------------------------------
# Directory path for the 'objects' dataset

base = r"C:\Users\ICL512\Desktop\semester 2 courses\reinforcement learning\am205_hw1_files\am205_hw1_files\problem5\objects\1600x1200\\"

# Read the regular well-lit image and normalize pixel values to [0, 1]
a = io.imread(base + "regular.png").astype(np.float64) / 255.0

# Read the three low-light images and normalize them
b = [io.imread(base + f"low{i+1}.png").astype(np.float64) / 255.0 for i in range(s)]

# Extract image dimensions
N, M, z = a.shape

#----------------------------------------------------------------
# Construct matrix A for least squares fitting
#----------------------------------------------------------------

# A will hold pixel values from low-light images and a bias term
# Each pixel contributes to 3 color channels across s images, plus one bias column
A = np.zeros((M * N, 3 * s + 1))

# Fill matrix A with RGB values from low-light images
for k in range(s):  # iterate over each low-light image
    for l in range(3):  # iterate over each color channel (R, G, B)
        c = b[k][:, :, l]
        A[:, k * 3 + l] = c.reshape(M * N)
A[:, 3 * s] = 1.0  # bias term

#----------------------------------------------------------------
# Construct target vector y from the regular image
#----------------------------------------------------------------

# y will hold the RGB values from the regular image
y = np.zeros((M * N, 3)) # initialize y
for l in range(3):
    y[:, l] = a[:, :, l].reshape(M * N) # fill y with RGB values

#----------------------------------------------------------------
# Solve the least squares problem to find coefficients F
#----------------------------------------------------------------

# F contains transformation coefficients (FB, FC, FD, Pconst)
F = np.linalg.lstsq(A, y, rcond=None)[0] # obtain coefficients
print("Fitting Coefficients (F):")
print(F)
#----------------------------------------------------------------
# Reconstruct the regular image (objects) using the fitted coefficients
#-----------------------------------------------------------------
ao = np.zeros((N, M, 3))
for l in range(3):
    c = np.dot(A, F[:, l])
    ao[:, :, l] = c.reshape(N, M)

# ----------------------------------------------------------------
# Compute pixel-wise difference (change)
# ----------------------------------------------------------------
# Change = Original - Reconstructed
change = a - ao

# Compute norm of the difference (overall reconstruction error)
norm_val = (np.linalg.norm(change.reshape(M * N * 3)))**2 / (M * N)
print("Norm (objects):", norm_val)

# ----------------------------------------------------------------
# Separate and visualize positive / negative components
# ----------------------------------------------------------------
# Positive: where original > reconstructed
# Negative: where reconstructed > original
pos_change = np.clip(change, 0, None)   # keep only positive differences
neg_change = np.clip(-change, 0, None)  # flip and keep only negative differences

# Scale differences by 10 for better visualization
scale_factor = 10.0
pos_scaled = np.clip(pos_change * scale_factor, 0, 1)
neg_scaled = np.clip(neg_change * scale_factor, 0, 1)

# Save visualizations
io.imsave("pos_diff_objects.png", (pos_scaled * 255).astype(np.uint8))
io.imsave("neg_diff_objects.png", (neg_scaled * 255).astype(np.uint8))

# ----------------------------------------------------------------
# Save reconstructed "objects" image
# ----------------------------------------------------------------
ao = np.clip(ao, 0, 1)
io.imsave("rec_image_objects.png", (ao * 255).astype(np.uint8))

# ----------------------------------------------------------------
# Process "bears" images using the same coefficients (F)
# ----------------------------------------------------------------
base = r"C:\Users\ICL512\Desktop\semester 2 courses\reinforcement learning\am205_hw1_files\am205_hw1_files\problem5\bears\1600x1200\\"

a = io.imread(base + "regular.png").astype(np.float64) / 255.0
b = [io.imread(base + f"low{i+1}.png").astype(np.float64) / 255.0 for i in range(s)]

# Update A with new data (bears low-light images)
for k in range(s): # iterate over each low-light image
    for l in range(3):
        c = b[k][:, :, l]
        A[:, k * 3 + l] = c.reshape(M * N) # fill A with new low-light data

# Reconstruct the regular image (bears)
ao = np.zeros((N, M, 3))
for l in range(3):
    c = np.dot(A, F[:, l])
    ao[:, :, l] = c.reshape(N, M)

# Compute reconstruction error for bears
diff = a - ao
norm_val = (np.linalg.norm(diff.reshape(M * N * 3)))**2 / (M * N)
print("Norm (bears):", norm_val)

# Save reconstructed image for bears
ao = np.clip(ao, 0, 1)
io.imsave("rec_image_bears.png", (ao * 255).astype(np.uint8))



