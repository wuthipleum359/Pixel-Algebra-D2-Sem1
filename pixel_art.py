import os
import numpy as np
from PIL import Image
import csv

data_folder = "D:\Python\Linear\Pixel Art Data\images"
summary_file = os.path.join(data_folder, "pixel_summary.csv")

if not os.path.exists(summary_file):
    with open(summary_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "filename", "scale", "n_pixels", "k", "mse"])


def mse(img1, img2):
    """คำนวณ Mean Squared Error ระหว่างสองภาพ"""
    v1 = img1.astype(np.float32).flatten()
    v2 = img2.astype(np.float32).flatten()
    return np.mean((v1 - v2) ** 2)

def resize_image(img, scale=0.1):
    """ใช้ Pillow แทน OpenCV ในการ resize"""
    h, w = img.shape[:2]
    new_w, new_h = int(w*scale), int(h*scale)
    pil_img = Image.fromarray(img)
    small = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(small)

def kmeans_numpy(pixels, k=16, max_iter=10):
    n_pixels = pixels.shape[0]
    np.random.seed(0)
    indices = np.random.choice(n_pixels, k, replace=False)
    centers = pixels[indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None] - centers[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([
            pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(centers, new_centers, atol=1e-2):
            break
        centers = new_centers

    return labels, centers.astype(np.uint8)

def pixel_art(img, scale=0.1, k=16):
    h, w = img.shape[:2]
    small = resize_image(img, scale=scale)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(k, pixels.shape[0])
    labels, centers = kmeans_numpy(pixels, k=k, max_iter=20)
    quantized = centers[labels].reshape(small.shape)
    pil_img = Image.fromarray(quantized)
    pixel_art_img = pil_img.resize((w, h), Image.NEAREST)
    return np.array(pixel_art_img)

input_folder = r"D:\Python\Linear\Pixel Art Data\images\new_project"
output_folder = r"D:\Python\Linear\Pixel Art Data\images\environment\environment_out"
category = "environment"

os.makedirs(output_folder, exist_ok=True)

threshold_mse = 300.0
scale_start = 0.01
scale_step = 0.005
ratio = 0.05
k_min = 4
k_max = 256

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        continue

    inp_path = os.path.join(input_folder, filename)
    img_name = os.path.splitext(filename)[0]
    img_output_dir = os.path.join(output_folder, img_name)
    os.makedirs(img_output_dir, exist_ok=True)

    img = np.array(Image.open(inp_path).convert("RGB"))

    scale = scale_start
    best_img = None
    best_distance = np.inf
    tempname = os.path.join(img_output_dir, f"{img_name}_log.txt")
    with open(tempname, "w", encoding='utf-8') as f:
        f.write(f"=== กำลังประมวลผล: {filename} ===\n")
    print(f"\n=== กำลังประมวลผล: {filename} ===")
    while scale <= 1:
        h, w = img.shape[:2]
        n_pixels = int(w * scale) * int(h * scale)
        k = int(n_pixels * ratio)
        k = max(k_min, min(k, k_max))

        art = pixel_art(img, scale=scale, k=k)
        distance = mse(img, art)
        with open(tempname, "a", encoding='utf-8') as f:
            f.write(f"scale={scale:.3f}, n_pixels={n_pixels}, k={k}, mse={distance:.4f}\n")
        print(f"scale={scale:.3f}, n_pixels={n_pixels}, k={k}, mse={distance:.4f}")
        with open(summary_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([category, filename, scale, n_pixels, k, distance])

        Image.fromarray(art).save(os.path.join(img_output_dir, f"pixel_art_{scale:.3f}.jpg"))

        if distance <= threshold_mse:
            best_img = art
            best_distance = distance
            break

        scale += scale_step

    if best_img is not None:
        out_path = os.path.join(img_output_dir, "result.jpg")
        Image.fromarray(best_img).save(out_path)
        with open(tempname, "a", encoding='utf-8') as f:
            f.write(f"{filename}: mse={best_distance:.4f}, scale={scale:.2f}, k={k}\n")
        print(f" {filename}: mse={best_distance:.4f}, scale={scale:.2f}, k={k}")
    else:
        print(f" {filename}: ไม่สามารถถึง threshold ที่กำหนดได้")


# ---------------- Main ----------------
# D:\Python\Linear\Pixel Art Data\images\plants\plants_out
# 
# "D:\Python\Linear\Pixel Art Data\images\people\people_out"
# 
# D:\Python\Linear\Pixel Art Data\images\food\food_out
# 
# D:\Python\Linear\Pixel Art Data\images\environment\environment_out
# 
# D:\Python\Linear\Pixel Art Data\images\buildings\buildings_out