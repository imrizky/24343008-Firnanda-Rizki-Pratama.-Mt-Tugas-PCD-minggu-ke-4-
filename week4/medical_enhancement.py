import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# =====================================================
# Medical Image Enhancement Function
# =====================================================
def medical_image_enhancement(medical_image, modality='X-ray'):

    report = {}

    # Normalisasi image ke 0–1
    img = medical_image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # =================================================
    # Enhancement berdasarkan modality
    # =================================================
    if modality == 'X-ray':

        p2, p98 = np.percentile(img, (2, 98))
        enhanced = np.clip((img - p2)/(p98 - p2), 0, 1)

        report["Modality"] = "X-ray"
        report["Technique"] = "Contrast Stretching"
        report["Clinical Goal"] = "Bone structure enhancement"

    elif modality == 'MRI':

        mean = np.mean(img)
        std = np.std(img)

        enhanced = (img - mean)/(std + 1e-6)
        enhanced = np.clip(enhanced, -2, 2)

        enhanced -= enhanced.min()
        enhanced /= enhanced.max()

        report["Modality"] = "MRI"
        report["Technique"] = "Intensity Normalization"
        report["Clinical Goal"] = "Soft tissue visibility"

    elif modality == 'CT':

        window_center = 0.5
        window_width = 0.5

        low = window_center - window_width/2
        high = window_center + window_width/2

        enhanced = np.clip((img - low)/(high-low), 0, 1)

        report["Modality"] = "CT"
        report["Technique"] = "Windowing"
        report["Clinical Goal"] = "Density visualization"

    elif modality == 'Ultrasound':

        enhanced = np.log1p(img)
        enhanced /= enhanced.max()

        report["Modality"] = "Ultrasound"
        report["Technique"] = "Log Compression"
        report["Clinical Goal"] = "Speckle noise reduction"

    else:
        enhanced = img
        report["Technique"] = "None"

    # =================================================
    # Metrics
    # =================================================
    report["Mean Intensity"] = float(np.mean(enhanced))
    report["Contrast (Std Dev)"] = float(np.std(enhanced))
    report["Min"] = float(np.min(enhanced))
    report["Max"] = float(np.max(enhanced))

    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced, report


# =====================================================
# LOAD IMAGE
# =====================================================
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "medical.jpg")

if not os.path.exists(image_path):
    print("ERROR: medical.jpg tidak ditemukan!")
    exit()

img = Image.open(image_path).convert("L")
medical_image = np.array(img)

print("Medical image loaded!")


# =====================================================
# PILIH MODALITY
# =====================================================
modality = "X-ray"
# pilihan:
# "X-ray", "MRI", "CT", "Ultrasound"


# =====================================================
# PROCESS
# =====================================================
enhanced_image, report = medical_image_enhancement(
    medical_image,
    modality
)


# =====================================================
# DISPLAY RESULT
# =====================================================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(medical_image, cmap='gray')
plt.title("Original Medical Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced_image, cmap='gray')
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# =====================================================
# PRINT REPORT
# =====================================================
print("\n=== Enhancement Report ===")
for key, value in report.items():
    print(f"{key}: {value}")