import rasterio
import os
import numpy as np

# 输入和输出目录
# input_dir = 'E:/The North America ecosystem carbon flux/Region_data/State_mosaci_geotifs/0216/Exclude_by_LC'
input_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Flux come/Extract_by_LC/'
output_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Flux come/Extract_by_LC/year/'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有TIFF文件
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开原始TIFF文件
        with rasterio.open(input_path) as src:
            data = src.read()  # 读取所有波段数据
            profile = src.profile  # 获取元数据
            nodata = src.nodata  # 获取Nodata值

            # 处理有效像元
            if nodata is not None:
                # 创建有效像元的掩码（排除Nodata）
                mask = (data != nodata)
            else:
                # 如果未设置Nodata，假设所有像素有效
                mask = np.ones_like(data, dtype=bool)

            # 转换为浮点型防止溢出
            processed_data = data.astype(np.float32)
            processed_data[mask] *= 365  # 仅有效像元乘以365

            # 更新元数据中的数据类型（确保与处理后的数据匹配）
            profile.update(dtype=np.float32, nodata=nodata)

            # 写入输出文件
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(processed_data)

print("处理完成！所有文件已保存至:", output_dir)