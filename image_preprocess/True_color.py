import os
from osgeo import gdal
import numpy as np

# 文件路径
band1_fn = 'E:/huapo/LC08_L2SP_128041_20240326_20240409_02_T1_SR_B4.TIF'  # 红波段
band2_fn = 'E:/huapo/LC08_L2SP_128041_20240326_20240409_02_T1_SR_B3.TIF'  # 绿波段
band3_fn = 'E:/huapo/LC08_L2SP_128041_20240326_20240409_02_T1_SR_B2.TIF'  # 蓝波段

# 读取红波段
in_ds_red = gdal.Open(band1_fn)
in_band_red = in_ds_red.GetRasterBand(1)

# 创建输出 TIFF 数据集
gtiff_driver = gdal.GetDriverByName('GTiff')
out_ds = gtiff_driver.Create(
    'E:/huapo/LC08_L2SP_128041_20240326_20240409.tiff',
    in_band_red.XSize,
    in_band_red.YSize,
    3,
    gdal.GDT_Byte  # 设置数据类型
)

# 设置投影和变换信息
out_ds.SetProjection(in_ds_red.GetProjection())
out_ds.SetGeoTransform(in_ds_red.GetGeoTransform())

# 读取红波段数据并写入输出数据集
out_band_red = out_ds.GetRasterBand(1)
out_band_red.WriteArray(in_band_red.ReadAsArray())

# 读取绿波段数据并写入输出数据集
in_ds_green = gdal.Open(band2_fn)
out_band_green = out_ds.GetRasterBand(2)
out_band_green.WriteArray(in_ds_green.ReadAsArray())

# 读取蓝波段数据并写入输出数据集
in_ds_blue = gdal.Open(band3_fn)
out_band_blue = out_ds.GetRasterBand(3)
out_band_blue.WriteArray(in_ds_blue.ReadAsArray())

# 计算统计数据
for i in range(1, 4):
    out_ds.GetRasterBand(i).ComputeStatistics(False)

# 创建金字塔概览
out_ds.BuildOverviews('average', [2, 4, 8, 16, 32])

# 保存数据
out_ds.FlushCache()

# 清理内存
del out_ds
del in_ds_red
del in_ds_green
del in_ds_blue

print("真彩色影像合成成功！")


