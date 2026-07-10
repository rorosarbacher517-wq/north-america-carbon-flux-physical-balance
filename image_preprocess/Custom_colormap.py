import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 定义值到颜色的映射
value_to_color = {
    1: '05450a', 2: '086a10', 3: '54a708', 4: '78d203',
    5: '009900', 6: 'c6b044', 7: 'dcd159', 8: 'dade48',
    9: 'fbff13', 10: 'b6ff05', 11: '27ff87', 12: 'c24f44',
    13: 'a5a5a5', 14: 'ff6d4c', 15: '69fff8', 16: 'f9ffa4',
    17: '1c0dff'
}

# 将十六进制颜色转换为RGB
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# 生成RGB颜色列表
color_map = np.array([hex_to_rgb(value_to_color[i]) for i in range(1, 18)])

# 读取.tif文件
with rasterio.open('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/clipped/clipped_Study_area_LC_Type.tif') as src:
    # 读取数据
    data = src.read(1)  # 假设地表覆盖图只有一个波段
    meta = src.meta.copy()

    # 创建输出图像
    height, width = data.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 应用颜色映射
    for value in range(1, 18):
        output_image[data == value] = color_map[value - 1]

    # 更新合并的元数据
    meta.update({
        'count': 3,  # 三个波段
        'dtype': 'uint8'
    })

# 保存新的.tif文件
with rasterio.open('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/clipped/clipped_Study_area_LC_Type_color.tif', 'w', **meta) as dst:
    dst.write(output_image[:, :, 0], 1)  # 写入红色通道
    dst.write(output_image[:, :, 1], 2)  # 写入绿色通道
    dst.write(output_image[:, :, 2], 3)  # 写入蓝色通道

print("颜色映射完成，文件已保存。")