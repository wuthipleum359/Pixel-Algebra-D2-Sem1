import pandas as pd
import numpy as np
import json

def calculate_elbow_point(points):
    """คำนวณหาจุด Elbow Point จากชุดข้อมูล (scale, mse)"""
    first_point = points[0]
    last_point = points[-1]
    
    line_vec = last_point - first_point
    if np.all(line_vec == 0): return points[len(points) // 2] # Handle case with one point
    
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = points - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    elbow_index = np.argmax(dist_to_line)
    return points[elbow_index]

# 1. โหลดข้อมูล
df = pd.read_csv(r"C:\Pleum101\Linear Algebra\model\pixel_summary.csv")

# 2. เตรียมเก็บผลลัพธ์
recommendations = []

# 3. วนลูปเพื่อหา Elbow Point ของทุกภาพ
for filename in df['filename'].unique():
    image_df = df[df['filename'] == filename].sort_values('scale').copy()
    
    if len(image_df) < 3:
        continue # ข้ามถ้าข้อมูลน้อยไป

    points = image_df[['scale', 'mse']].values
    
    # 4. คำนวณและเก็บผลลัพธ์
    recommended_point = calculate_elbow_point(points)
    recommended_mse = recommended_point[1]
    category = image_df['category'].iloc[0]
    
    recommendations.append({
        'category': category,
        'recommended_mse': recommended_mse
    })

# 5. สร้าง DataFrame จากผลลัพธ์ และหาค่าเฉลี่ย
rec_df = pd.DataFrame(recommendations)
avg_mse_by_category = rec_df.groupby('category')['recommended_mse'].mean().round(2).to_dict()

# 6. บันทึกเป็นไฟล์ JSON
output_filename = 'recommended_mse.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(avg_mse_by_category, f, indent=4)

print("สร้างไฟล์ค่า MSE แนะนำสำเร็จ!")
print(f"ผลลัพธ์ถูกบันทึกใน: {output_filename}")
print("\nค่าเฉลี่ยที่ได้:")
print(avg_mse_by_category)