import rasterio
import numpy as np
from skimage.morphology import binary_dilation, disk


def dilate_tif(input_tif, output_tif, dilation_size=2):
    # 读取输入TIF文件
    with rasterio.open(input_tif) as src:
        # 读取栅格数据
        data = src.read(1)  # 读取第一个波段
        profile = src.profile  # 获取元数据

        # 将数据转换为二值图像，条件为非零值
        binary_data = np.where(data > 0, 1, 0).astype(np.uint8)

        # 膨胀操作
        selem = disk(dilation_size)  # 创建一个圆形的结构元素
        dilated_data = binary_dilation(binary_data, selem)

        # 将膨胀后的数据转换为原始类型
        final_data = np.where(dilated_data > 0, 255, 0).astype(data.dtype)

    # 写入输出TIF文件
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(final_data, 1)


# 使用范例
input_tif_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/Exclude_lc_without_data/LC_11_13_16_17_Binarization.tif"  # 输入的TIF文件路径
output_tif_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/Exclude_lc_without_data/LC_11_13_16_17_Binarization_PZ_2_pixels.tif"  # 输出的TIF文件路径

dilate_tif(input_tif_path, output_tif_path, dilation_size=2)