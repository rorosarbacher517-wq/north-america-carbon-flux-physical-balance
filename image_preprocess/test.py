import numpy as np
from PIL import Image
import os
import rasterio

from PIL import Image
import os
import numpy as np
import rasterio

import os
import numpy as np
import gc
import datetime

a = np.log10(93)
print(a)

# a = 141 ,b = -19.360846215313593
#
# def ref_read_multiband(ref_doy_path):
#     # 读取并打开文件
#     with rasterio.open(ref_doy_path) as src:
#         # 读取所有波段到一个numpy数组中
#         data = src.read()
#         # 获取波段数和数组的维度
#         num_bands, height, width = data.shape
#         # 定义窗口大小
#         window_size = 3
#         # 计算输出数组的形状
#         output_shape = (height, width, num_bands, window_size, window_size)
#         # 重塑数组以创建滑动窗口视图
#         sub_arrays = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=data.itemsize * np.array(
#             [width * num_bands, num_bands, width * num_bands, num_bands, 1]))
#         lon_list, lat_list = [], []
#         for i in range(height):
#             for j in range(width):
#                 lon, lat = src.xy(i, j)
#                 lon_list.append(lon)
#                 lat_list.append(lat)
#
#         lon_lat_array = np.array(list(zip(lon_list, lat_list))).reshape((height, width, 2))
#         gc.collect()
#         return sub_arrays, lon_lat_array
#
#
# def read_multiband(ref_doy_path):
#     # 读取并打开文件
#     # Open the multi-band TIFF file
#     with rasterio.open(ref_doy_path) as src:
#         # Read all bands into a single numpy array
#         data = src.read()
#
#     # Get the number of bands and the dimensions of the array
#     num_bands, height, width = data.shape
#
#     # Define the window size
#     window_size = 3
#
#     # Calculate the shape of the output array
#     output_shape = (height, width, num_bands, window_size, window_size)
#
#     # Reshape the array to create the sliding window view
#     sub_arrays = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=data.itemsize * np.array(
#         [width * num_bands, num_bands, width * num_bands, num_bands, 1]))
#     # The sub_arrays variable now contains the data arranged in the required shape
#     gc.collect()
#     return sub_arrays
#
# # 将日期转为数字
# def out_day_by_date(Date):
#     '''
#     根据输入的日期计算该日期是在当年的第几天
#     '''
#     # 将字符串转为日期格式
#     Date = datetime.strptime(Date, '%Y-%m-%d').date()
#     # date = Date.apply(lambda x: datetime.strptime(x,'%Y%m%d'))
#     year = Date.year
#     month = Date.month
#     day = Date.day
#     months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
#     if 0 < month <= 12:
#         sum = months[month - 1]
#     else:
#         print("month error")
#     sum += day
#     leap = 0
#     # 接下来判断平年闰年
#     if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):  # and的优先级大于or
#         # 1、世纪闰年:能被400整除的为世纪闰年
#         # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
#         leap = 1
#     if (leap == 1) and (month > 2):
#         sum += 1  # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
#     return year,sum
#
#
# def pad_ndarray_to_uniform_shape(ndarray_list):
#     max_shape = tuple(np.max([arr.shape for arr in ndarray_list], axis=0))
#     padded_array_list = []
#     for arr in ndarray_list:
#         pad_widths = [(0, max_shape[i] - arr.shape[i]) for i in range(len(max_shape))]
#         padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)
#         padded_array_list.append(padded_arr)
#     return padded_array_list
#
# # 将每个站点&每一年的数据拼接起来组成一个输入数据
# # def concatenate_and_pad_time_series(ref_list):
# #     new_list = []
# #     max_shape = max(max(arr.shape[0] for arr in sublist) for sublist in ref_list)
# #     for sublist in ref_list:
# #         for arr in sublist:
# #             if arr.shape[0] < max_shape:
# #                 padding_shape = (max_shape - arr.shape[0],) + arr.shape[1:]
# #                 padding = np.full(padding_shape, -9999)
# #                 new_arr = np.concatenate((arr, padding), axis=0)
# #                 new_list.append(new_arr)
# #             else:
# #                 new_list.append(arr)
# #     combined_array = np.array(new_list)
# #     return combined_array
#
# def check_and_adjust_dimensions(block_ref_year_list):
#     expected_shape = (32, 32, 14, 3, 3)  # 期望的维度
#     different_dimensions_indices = []  # 存储不同维度的索引
#     for i, arr in enumerate(block_ref_year_list):
#         if arr.shape != expected_shape:
#             different_dimensions_indices.append(i)  # 记录不同维度的索引
#             block_ref_year_list[i] = np.zeros(expected_shape)  # 调整为相同的维度
#
#     print("具有不同维度的索引：", different_dimensions_indices)
#     return block_ref_year_list
#
# # 将list转为数组，并变换维度
# def transpose_reshape(block_ref_year_list):
#     # 在转换前统一block_ref_year_list的维度
#     # 检查是否list里的每个ndarray 维度都一样
#     # block_ref_year_list = check_and_adjust_dimensions(block_ref_year_list)
#     # block_ref_year_list_pad = pad_ndarray_to_uniform_shape(block_ref_year_list)
#     try:
#         print('开始转换')
#         block_year_array = np.asarray(block_ref_year_list)
#         print("转换成功")
#     except Exception as e:
#         print("转换失败:", e)
#     del block_ref_year_list
#     block_year_array_transpose = block_year_array.transpose(1, 2, 0, 3, 4,5)
#     # np.transpose(block_year_array, (1, 2, 0, 3, 4,5))
#     # reshape 将第0维和第1维乘起来，其余维度保持不变
#     block_year_array_reshape = block_year_array_transpose.reshape(block_year_array_transpose.shape[0]*block_year_array_transpose.shape[1],
#                                                                   block_year_array_transpose.shape[2],block_year_array_transpose.shape[3],
#                                                                  block_year_array_transpose.shape[4],block_year_array_transpose.shape[5])
#     gc.collect()
#     return block_year_array_reshape
#
#
# def process_product(product, date_list, blocks_path):
#     block_image_year_list = []
#     block_spatialtime_year_list = []
#
#     for date in date_list:
#         file_name = f'{product}_{date}.tif'
#         file_path = os.path.join(blocks_path, file_name)
#
#         if not os.path.isfile(file_path):
#             data = np.full((32, 32, 14 if product == 'MCD43A4' else 4 if product == 'MCD15A3H' else 6, 3, 3), -9999,
#                            dtype=np.float64)
#         else:
#             data = ref_read_multiband(file_path) if product == 'MCD43A4' else read_multiband(file_path)
#
#         if product == 'MCD43A4':
#             year, doy = out_day_by_date(date)
#             lon_lat_array = np.full((32, 32, 2), -9999, dtype=np.float64)
#             spatialtime_array = np.dstack((lon_lat_array, np.full((32, 32, 1), year),
#                                            np.full((32, 32, 1), doy)))
#             block_spatialtime_year_list.append(spatialtime_array.astype(np.float64))
#
#         block_image_year_list.append(data.astype(np.float64))
#
#     return block_image_year_list, block_spatialtime_year_list
#
# if __name__ == '__main__':
#     # 所有region 存储的根目录 每个国家的文件夹
#     region_base_dir = 'E:/Carbon_flux/data/regoin_image'
#
#     output_path = 'E:/Carbon_flux/data/regoin_image/region_images_test_blocks_match/'
#
#     nation_list = os.listdir(region_base_dir)
#     nation_path = os.path.join(region_base_dir, nation_list[4])
#
#     state_list = os.listdir(nation_path)
#     state_name = state_list[1]
#     state_path = os.path.join(nation_path, state_name)
#
#     output_state_path = os.path.join(output_path, state_name)
#     if not os.path.isdir(output_state_path):
#         os.makedirs(output_state_path)
#
#     blocks_list = os.listdir(state_path)
#     for k in range(34, len(blocks_list)):
#         block = blocks_list[k]
#         blocks_path = os.path.join(state_path, block)
#         output_block_path = os.path.join(output_state_path, block)
#
#         if not os.path.isdir(output_block_path):
#             os.makedirs(output_block_path)
#
#         product_list = ['MCD43A4', 'MCD15A3H', 'DAILY_AGGR']
#
#         for product in product_list:
#             ref_files_list = [file for file in os.listdir(blocks_path) if
#                               file.startswith(product) and file.endswith('.tif')]
#             date_list = [file.split('_')[1].split('.')[0] for file in ref_files_list]
#
#             block_image_year_list, block_spatialtime_year_list = process_product(product, date_list, blocks_path)
#
#             images_array = np.transpose(block_image_year_list)
#             np.save(output_block_path + '/' + block + '_' + product + '.npy', images_array)
#
#             if product == 'MCD43A4':
#                 spatialtime_array = np.asarray(block_spatialtime_year_list)
#                 block_year_flux_transpose = np.transpose(spatialtime_array, (1, 2, 0, 3))
#                 flux_array = block_year_flux_transpose.reshape(-1, block_year_flux_transpose.shape[2],
#                                                                block_year_flux_transpose.shape[3])
#                 np.save(output_block_path + '/' + block + '_spatialtime.npy', flux_array)
#
#             gc.collect()
#             print('The ' + product + ' have done')
#
#         print('The ' + block + ' have done')