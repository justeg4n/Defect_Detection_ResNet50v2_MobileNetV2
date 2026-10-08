import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all2\valid"
output_dir = os.path.join(base_dir, "noisy_images_texture_based")
os.makedirs(output_dir, exist_ok=True)

def compute_texture_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    # Laplacian variance = độ sắc nét / chi tiết
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    # Local entropy = độ phức tạp cục bộ
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return lap_var * entropy

for cls in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_path) or cls == "noisy_images_texture_based":
        continue

    print(f"\nProcessing class: {cls}")
    scores, paths = [], []
    for f in tqdm(os.listdir(cls_path)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            fp = os.path.join(cls_path, f)
            score = compute_texture_score(fp)
            if score is not None:
                scores.append(score)
                paths.append(fp)

    if len(scores) == 0:
        continue

    scores = np.array(scores)
    q3 = np.percentile(scores, 75)
    threshold = q3 * 1.5  # ảnh có độ biến thiên quá cao bị xem là nhiễu

    noisy_idx = np.where(scores > threshold)[0]
    noisy_folder = os.path.join(output_dir, cls)
    os.makedirs(noisy_folder, exist_ok=True)

    for i in noisy_idx:
        shutil.move(paths[i], os.path.join(noisy_folder, os.path.basename(paths[i])))

    print(f"Moved {len(noisy_idx)} noisy images out of {len(scores)} total")

print("\n=== DONE FILTERING ===")
