import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import joblib
import pandas as pd
import numpy as np
import cv2

def resize_image(img, scale=0.1):
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS))

def kmeans_numpy(pixels, k=16, max_iter=10):
    np.random.seed(0)
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    centers = pixels[indices]
    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None] - centers[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers, atol=1e-2): break
        centers = new_centers
    return labels, centers.astype(np.uint8)

def pixel_art(img, scale=0.1, k=16):
    h, w = img.shape[:2]
    small = resize_image(img, scale=scale)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(k, pixels.shape[0])
    labels, centers = kmeans_numpy(pixels, k=k, max_iter=20)
    quantized = centers[labels].reshape(small.shape)
    return np.array(Image.fromarray(quantized).resize((w, h), Image.NEAREST))

# ===================================================================
# 🎨 ส่วนของ UI (User Interface)
# ===================================================================

class PixelArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Generator")
        self.root.geometry("950x650")

        # ... (ตัวแปรต่างๆ เหมือนเดิม) ...
        self.image_path = None
        self.model_path = None
        self.features_path = None
        self.original_image = None
        self.result_image = None
        self.zoom_level = 1.0
        self.tk_result_image = None

        # ... (โค้ดสร้าง Frame ส่วนใหญ่เหมือนเดิม) ...
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        control_frame = tk.Frame(root, padx=10, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        image_frame = tk.Frame(root, padx=10, pady=10)
        image_frame.pack(expand=True, fill=tk.BOTH)

        model_frame = tk.LabelFrame(top_frame, text="1. เลือกไฟล์โมเดล")
        model_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.btn_select_model = tk.Button(model_frame, text="เลือกไฟล์โมเดล (.pkl)", command=self.select_model_files)
        self.btn_select_model.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_model_path = tk.Label(model_frame, text="Model: ยังไม่ได้เลือก", fg="grey")
        self.lbl_model_path.pack(side=tk.TOP, anchor="w")
        self.lbl_features_path = tk.Label(model_frame, text="Features: ยังไม่ได้เลือก", fg="grey")
        self.lbl_features_path.pack(side=tk.TOP, anchor="w")

        img_select_frame = tk.LabelFrame(control_frame, text="2. เลือกรูปภาพ")
        img_select_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.btn_select = tk.Button(img_select_frame, text="📁 เลือกไฟล์รูปภาพ", command=self.select_image_from_dialog)
        self.btn_select.pack(padx=5, pady=(5,2), fill=tk.X)

        # --- จุดแก้ไขที่ 1: เพิ่มปุ่มถ่ายรูป ---
        self.btn_camera = tk.Button(img_select_frame, text="📸 ถ่ายรูป", command=self.open_camera)
        self.btn_camera.pack(padx=5, pady=(2,5), fill=tk.X)

        self.lbl_path = tk.Label(img_select_frame, text="ยังไม่ได้เลือกไฟล์", fg="grey")
        self.lbl_path.pack(padx=5, pady=5)
        
        # ... (โค้ดส่วนที่เหลือใน __init__ เหมือนเดิม) ...
        settings_frame = tk.LabelFrame(control_frame, text="3. ตั้งค่า")
        settings_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        tk.Label(settings_frame, text="ประเภท (Category)").pack()
        self.categories = ['people', 'plants', 'food', 'environment', 'buildings']
        self.category_var = tk.StringVar(value=self.categories[0])
        self.category_menu = ttk.Combobox(settings_frame, textvariable=self.category_var, values=self.categories, state="readonly")
        self.category_menu.bind("<<ComboboxSelected>>", self.update_mse_for_category)
        self.category_menu.pack(padx=5, pady=2)
        tk.Label(settings_frame, text="ค่า MSE").pack()
        self.mse_slider = tk.Scale(settings_frame, from_=250, to=2250, orient=tk.HORIZONTAL, length=200, command=self.update_mse_label)
        self.mse_slider.pack(padx=5, pady=2)
        self.lbl_mse = tk.Label(settings_frame, text="")
        self.lbl_mse.pack(pady=2)
        self.update_mse_for_category()
        
        generate_frame = tk.LabelFrame(control_frame, text="4. สร้างภาพ")
        generate_frame.pack(side=tk.LEFT, padx=20, pady=5, fill=tk.Y)
        self.btn_generate = tk.Button(generate_frame, text="สร้างภาพ Pixel Art", command=self.generate_pixel_art, font=("Arial", 12, "bold"), bg="lightblue")
        self.btn_generate.pack(padx=10, pady=5, ipady=10, ipadx=10)

        result_actions_frame = tk.LabelFrame(control_frame, text="5. จัดการผลลัพธ์")
        result_actions_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.btn_save = tk.Button(result_actions_frame, text="💾 บันทึกผลลัพธ์", command=self.save_image)
        self.btn_save.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.btn_open_new = tk.Button(result_actions_frame, text="🖼️ เปิดในหน้าต่างใหม่", command=self.open_in_new_window)
        self.btn_open_new.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.panel_original = tk.Label(image_frame, text="ภาพต้นฉบับ", bg='grey90')
        self.panel_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.canvas_result = tk.Canvas(image_frame, bg='grey80', relief=tk.SUNKEN)
        self.canvas_result.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        self.canvas_result.create_text(200, 200, text="ภาพผลลัพธ์", anchor="center")
        self.canvas_result.bind("<MouseWheel>", self.zoom_image)
        self.canvas_result.bind("<Button-4>", self.zoom_image)
        self.canvas_result.bind("<Button-5>", self.zoom_image)

    # --- จุดแก้ไขที่ 2: สร้างฟังก์ชันสำหรับเปิดกล้อง ---
    def open_camera(self):
        """เปิดหน้าต่างกล้องด้วย OpenCV เพื่อถ่ายภาพ"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("ผิดพลาด", "ไม่สามารถเปิดกล้องได้")
            return

        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # เพิ่มข้อความแนะนำการใช้งานลงบนเฟรม
            cv2.putText(frame, "Press SPACE to capture, Q to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Camera - Press SPACE to capture', frame)
            key = cv2.waitKey(1)

            if key % 256 == ord('q'): # กด Q
                break
            elif key % 256 == 32: # กด Spacebar
                # บันทึกภาพชั่วคราว
                temp_path = "temp_capture.png"
                cv2.imwrite(temp_path, frame)
                self.load_image_from_path(temp_path) # โหลดภาพที่ถ่ายเข้ามาในโปรแกรม
                captured = True
                break
        
        # ปิดการเชื่อมต่อและหน้าต่าง
        cap.release()
        cv2.destroyAllWindows()
        
        if not captured:
             print("ยกเลิกการถ่ายภาพ")

    # --- จุดแก้ไขที่ 3: ปรับปรุงฟังก์ชันเลือกไฟล์ ---
    def select_image_from_dialog(self):
        """เปิด File Dialog เพื่อเลือกไฟล์ภาพ"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):
        """โหลดและแสดงผลภาพจาก Path ที่ระบุ (ใช้ร่วมกันระหว่างเลือกไฟล์และถ่ายภาพ)"""
        self.image_path = path
        self.lbl_path.config(text=os.path.basename(path), fg="black")
        
        # OpenCV อ่านภาพเป็น BGR, Pillow ใช้ RGB, ต้องแปลงสี
        if path.endswith("temp_capture.png"):
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(img_rgb)
        else:
            self.original_image = Image.open(self.image_path)
            
        self.display_image(self.original_image, self.panel_original)
        self.canvas_result.delete("all")
        self.canvas_result.create_text(200, 200, text="ภาพผลลัพธ์", anchor="center")
        self.result_image = None
        self.zoom_level = 1.0
        
    # เปลี่ยนชื่อฟังก์ชันเดิมเล็กน้อย
    def select_image(self):
        self.select_image_from_dialog()
        
    # ... (ฟังก์ชันที่เหลือเหมือนเดิมทั้งหมด) ...
    def update_mse_for_category(self, event=None):
        selected_category = self.category_var.get()
        if selected_category == 'food':
            self.mse_slider.set(701.62)
        elif selected_category == 'people':
            self.mse_slider.set(738.05)
        elif selected_category == 'plants':
            self.mse_slider.set(717.92)
        elif selected_category == 'environment':
            self.mse_slider.set(787.3)
        elif selected_category == 'buildings':
            self.mse_slider.set(832.91)
        self.update_mse_label(self.mse_slider.get())

    def update_mse_label(self, value):
        self.lbl_mse.config(text=f"MSE: {float(value):.1f}")

    def select_model_files(self, *args):
        model_p = filedialog.askopenfilename(title="เลือกไฟล์ Model (scale_predictor_model.pkl)", filetypes=[("PKL files", "*.pkl")])
        if not model_p: return
        features_p = filedialog.askopenfilename(title="เลือกไฟล์ Features (feature_columns.pkl)", filetypes=[("PKL files", "*.pkl")])
        if not features_p: return
        self.model_path = model_p
        self.features_path = features_p
        self.lbl_model_path.config(text=f"Model: {os.path.basename(self.model_path)}", fg="green")
        self.lbl_features_path.config(text=f"Features: {os.path.basename(self.features_path)}", fg="green")

    def display_image(self, img, panel):
        panel_w, panel_h = 400, 400
        try:
            panel_w = self.panel_original.winfo_width(); panel_h = self.panel_original.winfo_height()
        except: pass
        img_copy = img.copy()
        img_copy.thumbnail((panel_w - 20, panel_h - 20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_copy)
        panel.config(image=photo)
        panel.image = photo

    def display_result_on_canvas(self):
        if not self.result_image: return
        self.canvas_result.delete("all")
        w, h = self.result_image.size
        new_w = int(w * self.zoom_level); new_h = int(h * self.zoom_level)
        resized_img = self.result_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        self.tk_result_image = ImageTk.PhotoImage(resized_img)
        canvas_w = self.canvas_result.winfo_width(); canvas_h = self.canvas_result.winfo_height()
        self.canvas_result.create_image(canvas_w/2, canvas_h/2, anchor="center", image=self.tk_result_image)

    def zoom_image(self, event):
        if not self.result_image: return
        if event.num == 4 or event.delta > 0: self.zoom_level *= 1.2
        elif event.num == 5 or event.delta < 0: self.zoom_level /= 1.2
        self.zoom_level = max(0.1, min(self.zoom_level, 20.0))
        self.display_result_on_canvas()

    def generate_pixel_art(self):
        if not self.model_path or not self.features_path:
            messagebox.showwarning("Warning", "กรุณาเลือกไฟล์โมเดลก่อน"); return
        if not self.image_path:
            messagebox.showwarning("Warning", "กรุณาเลือกรูปภาพก่อน"); return
            
        desired_mse = self.mse_slider.get()
        category = self.category_var.get()
        
        try:
            model = joblib.load(self.model_path)
            feature_columns = joblib.load(self.features_path)
            img = np.array(Image.open(self.image_path).convert("RGB"))
            h, w = img.shape[:2]
            original_pixels = h * w

            input_data = pd.DataFrame({'mse': [desired_mse], 'original_pixels': [original_pixels]})
            input_data[f'category_{category}'] = 1
            input_data = input_data.reindex(columns=feature_columns, fill_value=0)
            predicted_scale = model.predict(input_data)[0]
            
            n_pixels = int(w * predicted_scale) * int(h * predicted_scale)
            k = max(4, min(int(n_pixels * 0.05), 256))
            result_array = pixel_art(img, scale=predicted_scale, k=k)
            
            self.result_image = Image.fromarray(result_array)
            self.zoom_level = 1.0
            self.display_result_on_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการสร้างภาพ:\n{e}")
            
    def save_image(self):
        if self.result_image is None:
            messagebox.showwarning("คำเตือน", "ยังไม่มีภาพผลลัพธ์ให้บันทึก")
            return
        original_filename = os.path.splitext(os.path.basename(self.image_path))[0]
        suggested_filename = f"pixelated_{original_filename}.png"
        filepath = filedialog.asksaveasfilename(
            initialfile=suggested_filename,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.result_image.save(filepath)
                messagebox.showinfo("สำเร็จ", f"บันทึกรูปภาพเรียบร้อยที่:\n{filepath}")
            except Exception as e:
                messagebox.showerror("ผิดพลาด", f"ไม่สามารถบันทึกไฟล์ได้:\n{e}")

    def open_in_new_window(self):
        if self.result_image is None:
            messagebox.showwarning("คำเตือน", "ยังไม่มีภาพผลลัพธ์ให้เปิด")
            return
        new_window = tk.Toplevel(self.root)
        new_window.title(f"ผลลัพธ์ - {os.path.basename(self.image_path)}")
        photo = ImageTk.PhotoImage(self.result_image)
        img_label = tk.Label(new_window, image=photo)
        img_label.image = photo 
        img_label.pack(padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtApp(root)
    root.mainloop()