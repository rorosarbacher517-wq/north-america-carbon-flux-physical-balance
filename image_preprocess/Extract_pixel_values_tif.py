# 按照3*3 的滑动窗口 读取 一个文件夹下的所有tif文件 然后叠加成一个array
import os
from osgeo import gdal
import numpy as np
import math
folder_path = '/home/fanbin/AMX_Carbon_flux/multiband_tif/2017-01-10'
output_path = '/home/fanbin/AMX_Carbon_flux/multiband_npy/2017-01-10'

# folder_path = 'E:/Carbon_flux/data/regoin_image/mcd43a4_tif_multiband/2017-05-15/'
# output_path = 'E:/Carbon_flux/data/regoin_image/mcd43a4_test_multiband_tif_npy/2017-05-15'

# **********************************
# 影像已经分区存储成tif文件 现需要依次读取每个分区内的所有波段的数据，每个波段以3*3窗口大小逐像素读取数据
for i in range(4):
    # 依次获取每个分块的所有波段数据
    block_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('block_'+str(i)+'.tif')]
    pixel_values_array = []
    window_size = 3
    # 遍历每个 TIFF 文件
    for tif_file in block_files:
        tiff_path = os.path.join(folder_path, tif_file)
        dataset = gdal.Open(tiff_path)
        # 读取数据
        raster = dataset.ReadAsArray()
        # 计算图像尺寸
        rows = raster.shape[0]
        cols = raster.shape[1]
        # 创建窗口视图
        windows = np.lib.stride_tricks.sliding_window_view(raster, (window_size, window_size))
        # 将窗口像元值数组添加到存储数组
        pixel_values_array.append(windows)
    # 获取并保存投影和仿射变换信息
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    with open(os.path.join(output_path, 'geotransform_block' + str(i) + '.txt'), 'w') as f:
        f.write(f"Projection: {projection}\n")
        f.write(f"GeoTransform: {geotransform}\n")
    # 将这个分块的多波段3*3窗口的数据叠加并转为多维array 输出保存为npy
    multiband_array = np.stack(pixel_values_array)
    # 变换一下维度 将第0维变换到最后一个维度去
    multiband_array_transformed = np.moveaxis(multiband_array, 0, -1)
    np.save(os.path.join(output_path, 'multibands_block'+str(i)+'.npy'), multiband_array_transformed)
    print('第'+str(i)+'个block处理完成')




# *************************************
# 先进行分块 然后在每个块里分别提取各个波段3*3大小的像素值
# # 设置窗口大小
# window_size = 3
#
# # 读取TIFF文件
# tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]
# tiff_path = os.path.join(folder_path, tif_files[0])
# dataset0 = gdal.Open(tiff_path)
#
# # 读取数据
# raster = dataset0.ReadAsArray()
# # 计算图像尺寸
# rows = raster.shape[0]
# cols = raster.shape[1]
# # 设置分块大小 将整个区域的tif平均划分为四个块
# block_size = 4
#
# # 对整个区域进行分块处理
# for i in range(0, rows, block_size):
#     for j in range(0, cols, block_size):
#         # 提取当前块的数据
#         block_data = raster[i:i + block_size, j:j + block_size]
#         # 依次处理每个波段的TIFF文件
#         pixel_values_array = []
#         for tif_file in tif_files:
#             tiff_path = os.path.join(folder_path, tif_file)
#             dataset = gdal.Open(tiff_path)
#             # 读取数据
#             raster_data = dataset.ReadAsArray()
#             # 提取3x3窗口像素值
#             windowed_array = raster_data[i:i + window_size, j:j + window_size]
#             pixel_values_array.append(windowed_array)
#
#         # 将所有波段的像素值数组叠合成多波段数组
#         multiband_array = np.stack(pixel_values_array, axis=-1)
#
#         # 保存多波段数组为.npy文件
#         output_file_name = f"block_{i}_{j}.npy"
#         output_file_path = os.path.join(output_path, output_file_name)
#         np.save(output_file_path, multiband_array)

# 整个块的所有tif_file 3*3窗口的影像输出保存成 块的编号.npy

# 先叠加多个波段的数据 再用窗口读取
# 读取文件夹下的所有波段数据并叠加成一个数组
# band_data = []
# for file in os.listdir(folder_path):
#     if file.endswith('.tif'):
#         tiff_path = os.path.join(folder_path, file)
#         dataset = gdal.Open(tiff_path)
#         band = dataset.GetRasterBand(1)
#         data = band.ReadAsArray()
#         band_data.append(data)
#
# # 将波段数据叠加成一个数组
# stacked_array = np.dstack(band_data)
# # 对数组以3x3大小的窗口读取
# window_size = 3
# rows, cols, num_bands = stacked_array.shape
# # 创建一个空数组，用于存储处理后的数据
# windowed_array = []
# # 设置每个块的大小
# block_size = 100  # 最大公约数math.gcd(rows, cols)
# # 逐块处理数据
# # 逐块处理数据
# # 重新计算块大小以确保能刚好将整个数据切分
# block_size_rows = rows - window_size + 1 if (rows - window_size + 1) % block_size == 0 else ((rows - window_size + 1) // block_size + 1) * block_size
# block_size_cols = cols - window_size + 1 if (cols - window_size + 1) % block_size == 0 else ((cols - window_size + 1) // block_size + 1) * block_size
#
# # 逐块处理数据
# for i in range(0, block_size_rows - window_size + 1, block_size):
#     for j in range(0, block_size_cols - window_size + 1, block_size):
#         # 读取数据块
#         block_data = stacked_array[i:i+block_size, j:j+block_size, :]
#
#         # 对块数据以3x3大小的窗口读取
#         block_rows, block_cols, num_bands = block_data.shape
#         block_windowed_array = np.zeros((block_rows - window_size + 1, block_cols - window_size + 1, window_size, window_size, num_bands))
#         for m in range(block_rows - window_size + 1):
#             for n in range(block_cols - window_size + 1):
#                 window = block_data[m:m+window_size, n:n+window_size, :]
#                 block_windowed_array[m, n, :, :, :] = window
#
#         # 将处理好的块数据添加到结果数组中
#         windowed_array.append(block_windowed_array)
#
# # 将结果数组转换为NumPy数组
# windowed_array = np.concatenate(windowed_array, axis=0)
#
# print(windowed_array.shape)  # 输出多维数组的形状

# # 将结果保存为 .npy格式
# np.save('E:/Carbon_flux/data/regoin_image/mcd43a4_test_multiband_tif_USAMainland_mask/npy/20170515.npy', windowed_array)
# #
# # 存储窗口大小
window_size = 3

# 读取TIFF文件
tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]

# # # 以3x3窗口大小读取每个像素的像元值并添加到数组
# # ***********************************************************************
# 分块处理
# 存储每个波段每个像素的3*3窗口内的像元值
# 存储每个波段每个像素的3*3窗口内的像元值
pixel_values_array = []

# 遍历每个 TIFF 文件
# 遍历每个 TIFF 文件
for tif_file in tif_files:
    tiff_path = os.path.join(folder_path, tif_file)
    dataset = gdal.Open(tiff_path)

    # 读取数据
    raster = dataset.ReadAsArray()

    # 计算图像尺寸
    rows = raster.shape[0]
    cols = raster.shape[1]

    # 创建窗口视图
    windows = np.lib.stride_tricks.sliding_window_view(raster, (window_size, window_size))

    # 将windows输出保存 所有波段叠加数据量太大了 因此单个波段单个波段的输出
    # import h5py
    # # 创建一个HDF5文件
    # with h5py.File(os.path.join(output_path,tif_file.replace('.tif','.h5')), 'w') as hf:
    #     # 将窗口数组存储在HDF5文件中
    #     hf.create_dataset('window_data', data=windows)
    # np.save(os.path.join(output_path,tif_file.replace('.tif','.npy')),windows)
    # 将窗口像元值数组添加到存储数组
    pixel_values_array.append(windows)

# 将所有波段的像素值list 转为array
# 所有数据叠加成一个占用内存太大了 会报错 Unable to allocate 208. GiB for an array with shape (16798, 26398, 3, 3, 14) and data type float32
# 因此在上面的时候就单个波段读取之后就进行存储
multiband_array = np.stack(pixel_values_array, axis=-1)
print(multiband_array.shape)


# # 遍历每个TIFF文件
# for tif_file in tif_files:
#     tiff_path = os.path.join(folder_path, tif_file)
#     dataset = gdal.Open(tiff_path)
#     raster = dataset.ReadAsArray()
#
#     # 使用numpy的as_strided函数创建窗口视图
#     shape = (raster.shape[0] - window_size + 1, raster.shape[1] - window_size + 1, window_size, window_size)
#     strides = (raster.strides[0], raster.strides[1], raster.strides[0], raster.strides[1])
#     windows = np.lib.stride_tricks.as_strided(raster, shape=shape, strides=strides)
#
#     # 将窗口视图转换为形状为 (n, window_size, window_size) 的数组
#     window_array = windows.reshape(-1, window_size, window_size) # 这样会报内存 因而需要分块处理
#
#     # 将窗口像元值数组添加到存储数组
#     pixel_values_array.append(window_array)
#
# # 将多个波段的窗口像元值数组转换为NumPy数组
# pixel_values_array = np.array(pixel_values_array)
# print(pixel_values_array.shape)  # 输出数组的形状

# # pixel_values_array = [] # 存储每个波段每个像素的3*3窗口内的像元值
# band_pixel_values_array = []
# # 遍历每个TIFF文件
# for tif_file in tif_files:
#     tiff_path = os.path.join(folder_path, tif_file)
#     dataset = gdal.Open(tiff_path)
#
#     # 获取地理转换信息
#     geotransform = dataset.GetGeoTransform()
#     x_origin = geotransform[0]
#     y_origin = geotransform[3]
#     pixel_width = geotransform[1]
#     pixel_height = geotransform[5]
#
#     # 计算图像尺寸
#     cols = dataset.RasterXSize
#     rows = dataset.RasterYSize
#
#     # 存储每个波段的窗口像元值数组
#     pixel_values_array = []
#
#     # 使用numpy的as_strided函数创建窗口视图并逐块处理
#     block_size = 1000  # 每次处理的行数
#     for i in range(0, rows, block_size):
#         for j in range(0, cols, block_size):
#             # 读取数据块
#             raster = dataset.ReadAsArray(j, i, block_size, block_size)
#
#             # 使用numpy的as_strided函数创建窗口视图
#             shape = (raster.shape[0] - window_size + 1, raster.shape[1] - window_size + 1, window_size, window_size)
#             strides = (raster.strides[0], raster.strides[1], raster.strides[0], raster.strides[1])
#             windows = np.lib.stride_tricks.as_strided(raster, shape=shape, strides=strides)
#
#             # 将窗口视图转换为形状为 (n, window_size, window_size) 的数组
#             window_array = windows.reshape(-1, window_size, window_size)
#
#             # 将窗口像元值数组添加到存储数组
#             pixel_values_array.append(window_array)
#
#     # 将每个波段的窗口像元值数组转换为NumPy数组
#     pixel_values_array = np.array(pixel_values_array)
#     print(pixel_values_array.shape)  # 输出数组的形状
#     band_pixel_values_array.append(pixel_values_array)
#
#
# # 将多个波段的窗口像元值数组转换为NumPy数组
# pixel_values_array = np.array(pixel_values_array)
# print(pixel_values_array.shape)  # 输出数组的形状


# # 打开第一个TIFF文件来获取地理转换信息
# first_tif_file = [file for file in os.listdir(folder_path) if file.endswith('.tif')][0]
# first_dataset = gdal.Open(os.path.join(folder_path, first_tif_file))
# geotransform = first_dataset.GetGeoTransform()
# x_origin = geotransform[0]
# y_origin = geotransform[3]
# pixel_width = geotransform[1]
# pixel_height = geotransform[5]
#
# # 获取每个像素的经纬度坐标
# pixel_coordinates_array = []
# rows, cols = first_dataset.RasterYSize, first_dataset.RasterXSize
# for i in range(rows):
#     for j in range(cols):
#         x = x_origin + j * pixel_width
#         y = y_origin + i * pixel_height
#         pixel_coordinates_array.append((x, y))
#
# # 将像素坐标数组转换为NumPy数组
# pixel_coordinates_array = np.array(pixel_coordinates_array)
#
