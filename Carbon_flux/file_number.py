import os

# 设置目标目录
target_dir = 'E:/Sentinel SAR/SAR_data/11SKA'

# 初始化子文件夹和tif文件的计数器
subfolders_count = 0
tif_files_count = 0

# 遍历目标目录中的所有文件和子目录
for root, dirs, files in os.walk(target_dir):
    for dir in dirs:
        subfolders_count += 1
    for file in files:
        if file.endswith('.tif'):
            tif_files_count += 1

# 打印结果
print(f"子文件夹数量: {subfolders_count}")
print(f"tif文件数量: {tif_files_count}")