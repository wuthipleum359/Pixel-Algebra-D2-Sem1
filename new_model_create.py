import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Step 0: ตั้งค่าโฟลเดอร์สำหรับบันทึกผล ---
output_folder = 'model_visualizations'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"สร้างโฟลเดอร์ '{output_folder}' สำหรับบันทึกแผนภาพแล้ว")

# --- Step 1: โหลดและเตรียมข้อมูล ---
print("\n--- Step 1: กำลังโหลดและเตรียมข้อมูล ---")
try:
    df_original = pd.read_csv(r'D:\Python\Linear\Pixel Art Data\images\pixel_summary.csv')
    print(f"โหลดข้อมูลสำเร็จ! จำนวน {df_original.shape[0]} แถว")
except FileNotFoundError:
    print("Error: ไม่พบไฟล์ 'pixel_summary.csv'")
    exit()

df = df_original.copy()

# --- Step 2: การทำ Feature Engineering ---
df = df[df['scale'] > 0].copy()
df['original_pixels'] = (df['n_pixels'] / (df['scale'] * df['scale'])).round().astype(int)
print("สร้างฟีเจอร์ 'original_pixels' เรียบร้อย")

# --- Step 3: (ใหม่) สร้าง Heatmap ความสัมพันธ์ของข้อมูลแต่ละประเภท ---
print("\n--- Step 2: กำลังสร้าง Heatmap ความสัมพันธ์ของข้อมูล ---")
categories = df_original['category'].unique().tolist()
numeric_cols = ['scale', 'n_pixels', 'k', 'mse', 'original_pixels']

for category in categories:
    plt.figure(figsize=(10, 8))
    
    category_df = df[df['category'] == category]
    corr_matrix = category_df[numeric_cols].corr()

    plt.rcParams['font.family'] = 'Tahoma'
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(f'Heatmap ความสัมพันธ์ของข้อมูลประเภท: {category}', fontsize=16, fontweight='bold')
    
    # บันทึกไฟล์
    save_path = os.path.join(output_folder, f'heatmap_correlation_{category}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() # ปิด plot เพื่อไม่ให้แสดงบนหน้าจอและคืน memory
    print(f"  - บันทึก Heatmap ของ '{category}' ที่ {save_path}")

# --- Step 4: การกำหนด Features, Target และเทรนโมเดล ---
df_dummies = pd.get_dummies(df, columns=['category'], prefix='', prefix_sep='')
y = df_dummies['scale']
X = df_dummies.drop(columns=['scale', 'filename', 'n_pixels', 'k'])
feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 3: กำลังเทรนโมเดล ---")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("เทรนโมเดลเสร็จสิ้น!")

# --- Step 5: ประเมินผลโมเดล ---
print("\n--- Step 4: กำลังประเมินผลโมเดล ---")
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f" - R-squared Score (R²): {r2:.4f}")

# --- Step 6: (ใหม่) สร้างและบันทึกกราฟเส้นทำนายผล (แยกไฟล์) ---
print("\n--- Step 5: กำลังสร้างและบันทึกกราฟเส้นทำนายผล ---")
test_df_viz = X_test.copy()
test_df_viz['scale_actual'] = y_test
test_df_viz['scale_predicted'] = predictions

plt.rcParams['font.family'] = 'Tahoma'

for category in categories:
    category_data = test_df_viz[test_df_viz[category] == 1]
    if not category_data.empty:
        plt.figure(figsize=(8, 6)) # สร้าง figure ใหม่ทุกครั้ง
        ax = sns.regplot(data=category_data, x='scale_actual', y='scale_predicted',
                         scatter_kws={'alpha':0.6, 's':50, 'edgecolor':'w'}, 
                         line_kws={'color':'#e74c3c', 'lw':2.5})
        ax.set_title(f"การทำนายผลของประเภท: {category}", fontweight='bold', fontsize=14)
        ax.set_xlabel("ค่า Scale จริง (Actual)", fontsize=12)
        ax.set_ylabel("ค่า Scale ที่ทำนาย (Predicted)", fontsize=12)
        ax.plot([0, 1], [0, 1], ls="--", c=".3") # เส้นอ้างอิง
        plt.tight_layout()
    
        save_path = os.path.join(output_folder, f'prediction_plot_{category}.png')
        plt.savefig(save_path, dpi=150)
        plt.close() # ปิด plot
        print(f"  - บันทึกกราฟทำนายผลของ '{category}' ที่ {save_path}")

print("\nการสร้างแผนภาพทั้งหมดเสร็จสิ้น!")