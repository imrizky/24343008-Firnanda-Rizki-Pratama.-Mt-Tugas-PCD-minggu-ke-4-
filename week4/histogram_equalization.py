import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

print("Working directory:", os.getcwd())
print("Isi folder:", os.listdir())


# =====================================================
# Manual Histogram Equalization
# =====================================================
def manual_histogram_equalization(image):

    image = image.astype(np.uint8)

    # 1. Histogram
    histogram = np.zeros(256)

    for pixel in image.flatten():
        histogram[pixel] += 1

    # 2. CDF
    cdf = np.cumsum(histogram)

    # 3. Transformation function
    cdf_min = cdf[np.nonzero(cdf)][0]
    total_pixels = image.size

    transform = np.round(
        (cdf - cdf_min) /
        (total_pixels - cdf_min) * 255
    ).astype(np.uint8)

    # 4. Apply mapping
    equalized_image = transform[image]

    return equalized_image, histogram


# =====================================================
# LOAD IMAGE
# =====================================================
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "input.jpg")

# cek file ada atau tidak
if not os.path.exists(image_path):
    print("ERROR: File input.jpg tidak ditemukan!")
    print("Letakkan gambar di folder yang sama.")
    exit()

img = Image.open(image_path).convert("L")
gray_image = np.array(img)

print("Image loaded successfully!")
print("Shape:", gray_image.shape)


# =====================================================
# PROCESS
# =====================================================
equalized_img, hist_before = manual_histogram_equalization(gray_image)

# histogram setelah
hist_after = np.zeros(256)
for pixel in equalized_img.flatten():
    hist_after[pixel] += 1


# =====================================================
# DISPLAY RESULT
# =====================================================
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(hist_before)
plt.title("Histogram Before")

plt.subplot(2,2,4)
plt.plot(hist_after)
plt.title("Histogram After")

plt.tight_layout()
plt.show()