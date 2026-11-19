import joblib
import os
import pandas as pd
import numpy as np
from PIL import Image

# ===================================================================
# สำคัญ: ต้องนำฟังก์ชันสร้าง Pixel Art ที่คุณใช้ มาไว้ที่นี่ด้วย
# เพื่อให้สามารถสร้างภาพจาก scale ที่ทำนายได้
# ===================================================================

def resize_image(img, scale=0.1):
    """ใช้ Pillow ในการ resize"""
    h, w = img.shape[:2]
    new_w, new_h = int(w*scale), int(h*scale)
    pil_img = Image.fromarray(img)
    small = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(small)

def kmeans_numpy(pixels, k=16, max_iter=10):
    """K-means ที่เขียนใหม่ด้วย numpy"""
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
    """สร้างภาพ pixel art จากภาพต้นฉบับ"""
    h, w = img.shape[:2]
    small = resize_image(img, scale=scale)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(k, pixels.shape[0])
    labels, centers = kmeans_numpy(pixels, k=k, max_iter=20)
    quantized = centers[labels].reshape(small.shape)
    pil_img = Image.fromarray(quantized)
    pixel_art_img = pil_img.resize((w, h), Image.NEAREST)
    return np.array(pixel_art_img)

# ===================================================================
# ⚙️ ส่วนหลักในการทำนายผล
# ===================================================================

def get_pixel_art_from_mse(image_path, category, desired_mse):
    """
    ฟังก์ชันหลัก: รับภาพ, ประเภท, และ MSE ที่ต้องการ
    แล้วคืนค่าเป็นภาพ Pixel Art ที่สร้างขึ้น
    """
    # --- 1. โหลดโมเดลที่เทรนไว้ ---
    try:
        model = joblib.load(r"C:\Pleum101\Linear Algebra\model\scale_predictor_model.pkl")
        # *** Best Practice: โหลดลิสต์ของ feature columns ที่ใช้ตอนเทรนมาด้วย
        # เพื่อป้องกันความผิดพลาดเรื่องลำดับของคอลัมน์
        feature_columns = joblib.load(r"C:\Pleum101\Linear Algebra\model\feature_columns.pkl")
    except FileNotFoundError:
        print("Error: ไม่พบไฟล์โมเดล 'scale_predictor_model.pkl' หรือ 'feature_columns.pkl'")
        print("กรุณารันสคริปต์สำหรับเทรนโมเดลก่อน")
        return None

    # --- 2. เตรียมข้อมูล Input สำหรับภาพใหม่ ---
    # โหลดภาพเพื่อเอาขนาด original
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    original_pixels = h * w

    # สร้าง DataFrame สำหรับทำนาย (ต้องมีทุกคอลัมน์เหมือนตอนเทรน)
    input_data = pd.DataFrame({
        'mse': [desired_mse],
        'original_pixels': [original_pixels]
    })
    
    # One-Hot Encode สำหรับ category ของภาพใหม่
    input_data[f'category_{category}'] = 1

    # *** ขั้นตอนที่สำคัญที่สุด: เรียงคอลัมน์ให้เหมือนตอนเทรนเป๊ะๆ ***
    # reindex จะสร้างคอลัมน์ที่ขาดไป (เช่น category อื่นๆ) และเติม 0 ให้เอง
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    print("ข้อมูล Input ที่จะใช้ทำนาย:")
    print(input_data)

    # --- 3. ทำนายผล (Predict) ---
    predicted_scale = model.predict(input_data)
    
    # ผลลัพธ์ที่ได้จะเป็น array, เราจึงดึงค่าแรกออกมา
    final_scale = predicted_scale[0]
    
    print(f"\n🚀 ผลการทำนาย 🚀")
    print(f"เพื่อให้ได้ MSE ประมาณ {desired_mse:.2f}")
    print(f"โมเดลแนะนำให้ใช้ Scale: {final_scale:.4f}")

    # --- 4. นำผลลัพธ์ไปใช้งาน ---
    # คำนวณค่า k ที่เหมาะสมสำหรับ scale ใหม่
    n_pixels = int(w * final_scale) * int(h * final_scale)
    k = max(4, min(int(n_pixels * 0.05), 256))

    # สร้างภาพ Pixel Art ด้วย scale ที่ทำนายได้
    final_image = pixel_art(img, scale=final_scale, k=k)
    
    return final_image

# ===================================================================
# ✨ ตัวอย่างการเรียกใช้งาน
# ===================================================================
if __name__ == "__main__":
    # -- ตั้งค่าสำหรับภาพใหม่ที่คุณต้องการ --
    NEW_IMAGE_PATH = r"C:\Users\wuthi\Downloads\michael-sum-LEpfefQf4rU-unsplash.webp" # <--- แก้ไข path รูปภาพของคุณ
    IMAGE_CATEGORY = "people"                               # <--- แก้ไขประเภทของภาพ
    DESIRED_MSE = 2000.0                                     # <--- กำหนด MSE ที่คุณต้องการ

    # -- เรียกใช้ฟังก์ชัน --
    if os.path.exists(NEW_IMAGE_PATH):
        result_image_array = get_pixel_art_from_mse(NEW_IMAGE_PATH, IMAGE_CATEGORY, DESIRED_MSE)

        # -- แสดงผลหรือบันทึกภาพ --
        if result_image_array is not None:
            result_pil_image = Image.fromarray(result_image_array)
            result_pil_image.show() # เปิดแสดงภาพ
            
            # หรือบันทึกเป็นไฟล์
            # output_path = "predicted_pixel_art.png"
            # result_pil_image.save(output_path)
            # print(f"บันทึกภาพผลลัพธ์ที่: {output_path}")

    else:
        print(f"Error: ไม่พบไฟล์ที่ '{NEW_IMAGE_PATH}'")