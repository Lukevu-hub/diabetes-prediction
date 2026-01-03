import kagglehub
import shutil
import os

# 1. Download latest version (Tải bản mới nhất về thư mục cache của hệ thống)
raw_path = kagglehub.dataset_download("imtkaggleteam/diabetes")

print("Path to dataset files:", raw_path)

# 2. Senior Step: Move data to project folder (Di chuyển về folder dự án)
# Chúng ta muốn dữ liệu nằm trong folder 'data/' để dễ quản lý và deploy sau này.
destination_folder = 'data'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Duyệt qua các file trong thư mục cache và copy về folder 'data'
for filename in os.listdir(raw_path):
    file_path = os.path.join(raw_path, filename)
    if os.path.isfile(file_path):
        shutil.copy(file_path, destination_folder)
        print(f"Moved: {filename} to {destination_folder}/")