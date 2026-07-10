#

import os
import numpy as np
import rasterio
from datetime import datetime
import gc
import psutil


def array_expand_multiband(array, expand_size):
    # 获取原始数组的形状
    num_bands, height, width = array.shape

    # 创建新数组进行扩展
    expanded_array = np.zeros((num_bands, height + expand_size + 1, width + expand_size + 1), dtype=array.dtype)

    # 复制原数组到新数组的内部区域
    expanded_array[:, 1:-1, 1:-1] = array

    # 将原数组的上下左右四周进行扩展
    expanded_array[:, 0, 1:-1] = array[:, 0, :]  # 复制第一行
    expanded_array[:, -1, 1:-1] = array[:, -1, :]  # 复制最后一行
    expanded_array[:, 1:-1, 0] = array[:, :, 0]  # 复制第一列
    expanded_array[:, 1:-1, -1] = array[:, :, -1]  # 复制最后一列
    expanded_array[:, 0, 0] = array[:, 0, 0]  # 左上角像素
    expanded_array[:, 0, -1] = array[:, 0, -1]  # 右上角像素
    expanded_array[:, -1, 0] = array[:, -1, 0]  # 左下角像素
    expanded_array[:, -1, -1] = array[:, -1, -1]  # 右下角像素
    return expanded_array

def ref_read_multiband(ref_doy_path):
    # 读取并打开文件
    with rasterio.open(ref_doy_path) as src:
        # 读取所有波段到一个numpy数组中
        data = src.read()  # Print an example band for reference
        data[2, :, :]  # data.shape = 6,32,32
        # 获取波段数和数组的维度
        num_bands, height, width = data.shape  # height =32,width =32,num_bands =6
        # Define window size
        window_size = 3
        # Pad the data array to handle edges properly
        # padded_data = np.pad(data, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=-9999)  # 6,34,34
        # padded_data[2, :, :]
        padded_data = array_expand_multiband(data, 1)
        padded_data[1, :, :]
        # Create a view with the desired shape
        shape = (height, width, num_bands, window_size, window_size)
        strides = (padded_data.strides[1], padded_data.strides[2], padded_data.strides[0], padded_data.strides[1],
                   padded_data.strides[2])  # padded_data.strides[0]、padded_data.strides[1] 和 padded_data.strides[2] 分别代表了 padded_data 数组在第一个维度（通常是波段数）、第二个维度（通常是高度）和第三个维度（通常是宽度）上相邻元素之间的字节数
        sub_arrays = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
        # print(sub_arrays[1, 1, 0, :, :])
        lon_list, lat_list = [], []
        for i in range(height):
            for j in range(width):
                lon, lat = src.xy(i, j)
                lon_list.append(lon)
                lat_list.append(lat)
        lon_lat_array = np.array(list(zip(lon_list, lat_list))).reshape((height, width, 2))
        gc.collect()
        return sub_arrays,lon_lat_array

def read_multiband(ref_doy_path):
    # 读取并打开文件
    with rasterio.open(ref_doy_path) as src:
        # 读取所有波段到一个numpy数组中
        data = src.read() # Print an example band for reference
        data[2,:,:] # data.shape = 6,32,32
        # 获取波段数和数组的维度
        num_bands, height, width = data.shape   # height =32,width =32,num_bands =6
        # Define window size
        window_size = 3
        # Pad the data array to handle edges properly
        # padded_data = np.pad(data, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=-9999) # 6,34,34
        # 获取原始数组的边缘值
        padded_data = array_expand_multiband(data, 1)
        padded_data[2, :, :]
        # Create a view with the desired shape
        shape = (height, width, num_bands, window_size, window_size)
        strides = (padded_data.strides[1], padded_data.strides[2], padded_data.strides[0], padded_data.strides[1],
                   padded_data.strides[2])  # padded_data.strides[0]、padded_data.strides[1] 和 padded_data.strides[2] 分别代表了 padded_data 数组在第一个维度（通常是波段数）、第二个维度（通常是高度）和第三个维度（通常是宽度）上相邻元素之间的字节数
        sub_arrays = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
        sub_arrays[0,0,2,:,:]
    # The sub_arrays variable now contains the data arranged in the required shape
    gc.collect()
    return sub_arrays

# 将日期转为数字
def out_day_by_date(Date):
    '''
    根据输入的日期计算该日期是在当年的第几天
    '''
    # 将字符串转为日期格式
    Date = datetime.strptime(Date, '%Y-%m-%d').date()
    # date = Date.apply(lambda x: datetime.strptime(x,'%Y%m%d'))
    year = Date.year
    month = Date.month
    day = Date.day
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if 0 < month <= 12:
        sum = months[month - 1]
    else:
        print("month error")
    sum += day
    leap = 0
    # 接下来判断平年闰年
    if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):  # and的优先级大于or
        # 1、世纪闰年:能被400整除的为世纪闰年
        # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
        leap = 1
    if (leap == 1) and (month > 2):
        sum += 1  # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    return year,sum


def pad_ndarray_to_uniform_shape(ndarray_list):
    max_shape = tuple(np.max([arr.shape for arr in ndarray_list], axis=0))
    padded_array_list = []
    for arr in ndarray_list:
        pad_widths = [(0, max_shape[i] - arr.shape[i]) for i in range(len(max_shape))]
        padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)
        padded_array_list.append(padded_arr)
    return padded_array_list


def check_and_adjust_dimensions(block_ref_year_list):
    expected_shape = (32, 32, 14, 3, 3)  # 期望的维度
    different_dimensions_indices = []  # 存储不同维度的索引
    for i, arr in enumerate(block_ref_year_list):
        if arr.shape != expected_shape:
            different_dimensions_indices.append(i)  # 记录不同维度的索引
            block_ref_year_list[i] = np.zeros(expected_shape)  # 调整为相同的维度
    print("具有不同维度的索引：", different_dimensions_indices)
    return block_ref_year_list

# 将list转为数组，并变换维度
def transpose_reshape(block_ref_year_list):
    # 在转换前统一block_ref_year_list的维度
    # 检查是否list里的每个ndarray 维度都一样
    # block_ref_year_list = check_and_adjust_dimensions(block_ref_year_list)
    # block_ref_year_list_pad = pad_ndarray_to_uniform_shape(block_ref_year_list)
    try:
        block_year_array = np.asarray(block_ref_year_list)
    except Exception as e:
        print("转换失败:", e)
    del block_ref_year_list
    # 变换维度 source:要移动的轴的原始位置。destination：每个原始轴的目标位置。
    block_year_array_transpose = np.moveaxis(block_year_array, [0, 1, 2], [2, 0, 1])
    # block_year_transpose = np.transpose(block_year_array, (1, 2, 0, 3, 4, 5))
    block_year_array_transpose[:,:,0,0,1,1]
    # reshape 将第0维和第1维乘起来，其余维度保持不变
    block_year_array_reshape = block_year_array_transpose.reshape(block_year_array_transpose.shape[0]*block_year_array_transpose.shape[1],
                                                                  block_year_array_transpose.shape[2],block_year_array_transpose.shape[3],
                                                                 block_year_array_transpose.shape[4],block_year_array_transpose.shape[5])
    block_year_array_reshape[:, :, 0, 1, 1]
    gc.collect()
    return block_year_array_reshape


def multi_variables_match(blocks_path,output_block_path):
    block_ref_year_list = []
    block_lai_year_list = []
    block_mete_year_list = []
    block_spatialtime_year_list = []
    # 获取ref文件 以MCD43A4开头，日期结尾；e.g. MCD43A4_2017-01-01
    # 依次遍历以MCD43A4开头的ref文件，同时得到ref的日期
    ref_files_list = [file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif')]
    # for k in range(0, len(ref_files_list)):
    # 使用生成器表达式
    # ref_files = (file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif'))
    # for ref_filename in ref_files:
    for f in range(0,len(ref_files_list)):
    # for f in range(0, 11):
        ref_file_name = ref_files_list[f]
        # 得到ref文件的路径
        ref_doy_path = os.path.join(blocks_path, ref_file_name)
        ref_doy_path = ref_doy_path.replace('\\', '/')
        # 获取文件名
        if not os.path.isfile(ref_doy_path):
            ref_datasets= np.full((32,32,14, 3, 3), -9999)
            lon_lat_array = np.full((32,32,2), -9999)
        else:
            # 再按3*3滑动窗口大小读取这个数据
            ref_datasets, lon_lat_array = ref_read_multiband(ref_doy_path)

        # ref_datasets = np.array(ref_datasets, dtype=np.float)
        ref_datasets = ref_datasets.astype(np.float64)
        block_ref_year_list.append(ref_datasets)
        # # 再按3*3滑动窗口大小读取这个数据
        # ref_datasets, lon_lat_array = ref_read_multiband(ref_doy_path)
        # block_ref_year_list.append(ref_datasets)
        # 提取日期信息
        ref_file_str = ref_file_name.replace('.tif', '')  # 移除文件扩展名
        ref_date = ref_file_str.split('_')[1]  # 以'_'分割文件名
        # 返回pixel_index,lat,lon,year,doy
        # 返回影像的每个像素的位置索引和中心经纬度坐标
        year, doy = out_day_by_date(ref_date)
        # 将year和doy填充到lon_lat_array中  变成 32，32，4 （4分别代表这个像素的lon，lat,year,doy）
        # 在lon_lat_array(32, 32, 2)中填充year和doy，变成 (32, 32, 4)
        spatialtime_array = np.dstack((
                                      lon_lat_array, np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), year),
                                      np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), doy)))
        # 添加每个像素的空间位置和时间信息
        spatialtime_array = spatialtime_array.astype(np.float64)
        # spatialtime_array = np.array(spatialtime_array, dtype=np.float)
        block_spatialtime_year_list.append(spatialtime_array)
        ## 加载lai数据
        lai_filename = 'MCD15A3H_' + ref_date + '.tif'
        lai_doy_path = os.path.join(blocks_path, lai_filename)
        lai_doy_path = lai_doy_path.replace('\\', '/')
        if not os.path.isfile(lai_doy_path):
            lai_datasets = np.full((32, 32, 4, 3, 3), -9999)
        else:
            lai_datasets = read_multiband(lai_doy_path)
        lai_datasets = lai_datasets.astype(np.float64)
        block_lai_year_list.append(lai_datasets)

        # 加载meteorology数据
        mete_filename = 'DAILY_AGGR_' + ref_date + '.tif'
        mete_doy_path = os.path.join(blocks_path, mete_filename)
        mete_doy_path = mete_doy_path.replace('\\', '/')
        if not os.path.isfile(mete_doy_path):
            mete_datasets = np.full((32, 32, 6, 3, 3), -9999)
        else:
            mete_datasets = read_multiband(mete_doy_path)
        mete_datasets = mete_datasets.astype(np.float64)
        block_mete_year_list.append(mete_datasets)
    # 将list转为数组,维度变换 n,366,b,3,3
    mem = psutil.virtual_memory()
    print(mem.percent)
    ref_array = transpose_reshape(block_ref_year_list)
    lai_array = transpose_reshape(block_lai_year_list)
    mete_array = transpose_reshape(block_mete_year_list)
    spatialtime_array = np.asarray(block_spatialtime_year_list)
    # 将数组维度变为（32，32，365，3，3）
    block_year_flux_transpose = np.moveaxis(spatialtime_array, [0, 1, 2], [2, 0, 1])
    # reshape
    flux_array = block_year_flux_transpose.reshape(
        block_year_flux_transpose.shape[0] * block_year_flux_transpose.shape[1],
        block_year_flux_transpose.shape[2], block_year_flux_transpose.shape[3])

    # 将这几个array输出保存为.npy格式
    np.save(output_block_path + '/' + block + '_ref.npy', ref_array)
    np.save(output_block_path + '/' + block + '_lai.npy', lai_array)
    np.save(output_block_path + '/' + block + '_mete.npy', mete_array)
    np.save(output_block_path + '/' + block + '_spatialtime.npy', flux_array)

    # 释放内存
    del ref_array, lai_array, mete_array, spatialtime_array, block_year_flux_transpose, flux_array,block_ref_year_list,\
        block_lai_year_list,\
        block_mete_year_list,\
        block_spatialtime_year_list,
    gc.collect()
    print('The ' + block + ' have done')

if __name__ == '__main__':
    # 所有region 存储的根目录 每个国家的文件夹
    region_base_dir = 'E:/Paper code/code/Vegetation_productivity_prediction/Region_prediction/data/'
    # "/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/images_split_to_blocks/"

    output_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Region_prediction/data/blocks_variables_match/'
    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_base_dir)
    nation_path = os.path.join(region_base_dir, 'images_split_to_blocks')

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    for state in state_list:
        state_name = state
        state_path = os.path.join(nation_path,state_name)
        output_state_path = os.path.join(output_path,state_name)
        if not os.path.isdir(output_state_path):
            os.makedirs(output_state_path)
        # 进入每个州下面的blocks目录
        blocks_list = os.listdir(state_path)
        for k in range(0,len(blocks_list)): # len(blocks_list)
            block = blocks_list[k]
            blocks_path = os.path.join(state_path, block)
            output_block_path = os.path.join(output_state_path, block)
            if not os.path.isdir(output_block_path):
                os.makedirs(output_block_path)
            product_spatialtime_filename = output_block_path + '/' + block + '_spatialtime.npy'
            if os.path.exists(product_spatialtime_filename):
                continue
            product_list = ['MCD43A4','MCD15A3H','DAILY_AGGR'] # 'MCD43A4','MCD15A3H',,'MCD15A3H','DAILY_AGGR'
            # multi_variables_match(blocks_path,output_block_path)
            # 获取ref文件
            # ref_files = (file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif'))
            ref_files_list = [file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif')]
            # 再从这些files里获取日期list
            date_list = [file.split('_')[1].split('.')[0] for file in ref_files_list]
            del ref_files_list
            for p in range(0,len(product_list)):
                product = product_list[p]
                block_image_year_list = []
                block_spatialtime_year_list = []
                product_filename = output_block_path + '/' + block + '_' + product + '.npy'
                if os.path.exists(product_filename):
                    continue

                for i in range(0,len(date_list)):
                    file_name = product +'_'+date_list[i]+'.tif'
                    file_path = os.path.join(blocks_path,file_name)
                    if product == 'MCD43A4':
                        # 获取文件名
                        if not os.path.isfile(file_path):
                            ref_datasets = np.full((32, 32, 14, 3, 3), -9999,dtype='float64')
                            lon_lat_array = np.full((32, 32, 2), -9999,dtype='float64')
                        else:
                            # 再按3*3滑动窗口大小读取这个数据
                            ref_datasets, lon_lat_array = ref_read_multiband(file_path)
                        # ref_datasets = np.array(ref_datasets, dtype=np.float)
                        ref_datasets = ref_datasets.astype(np.float64)
                        # ref_datasets[31,31,0,:,:]
                        block_image_year_list.append(ref_datasets)
                        # # 再按3*3滑动窗口大小读取这个数据
                        year, doy = out_day_by_date(date_list[i])
                        # 将year和doy填充到lon_lat_array中  变成 32，32，4 （4分别代表这个像素的lon，lat,year,doy）
                        # 在lon_lat_array(32, 32, 2)中填充year和doy，变成 (32, 32, 4)
                        spatialtime_array = np.dstack((
                            lon_lat_array, np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), year),
                            np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), doy)))
                        # 添加每个像素的空间位置和时间信息
                        spatialtime_array = spatialtime_array.astype(np.float64)
                        # spatialtime_array = np.array(spatialtime_array, dtype=np.float)
                        block_spatialtime_year_list.append(spatialtime_array)
                    elif product == 'MCD15A3H':
                        if not os.path.isfile(file_path):
                            lai_datasets = np.full((32, 32, 4, 3, 3), -9999,dtype='float64')
                        else:
                            lai_datasets = read_multiband(file_path)
                        lai_datasets = lai_datasets.astype(np.float64)
                        block_image_year_list.append(lai_datasets)
                    elif product == 'DAILY_AGGR':
                        if not os.path.isfile(file_path):
                            mete_datasets = np.full((32, 32, 6, 3, 3), -9999,dtype='float64')
                        else:
                            mete_datasets = read_multiband(file_path)
                        mete_datasets = mete_datasets.astype(np.float64)
                        block_image_year_list.append(mete_datasets)
                images_array = transpose_reshape(block_image_year_list)
                # print(images_array.shape)
                images_array[:,:,2,1,1]
                # 将这几个array输出保存为.npy格式
                np.save(output_block_path + '/' + block + '_' + product + '.npy', images_array)
                if product == 'MCD43A4':
                    spatialtime_array = np.asarray(block_spatialtime_year_list)
                    # 将数组维度变为（32，32，365，3，3）
                    block_year_flux_transpose = np.moveaxis(spatialtime_array, [0, 1, 2], [2, 0, 1])
                    # block_year_flux_transpose1 = np.transpose(spatialtime_array, (2, 0, 1))
                    # reshape
                    flux_array = block_year_flux_transpose.reshape(
                        block_year_flux_transpose.shape[0] * block_year_flux_transpose.shape[1],
                        block_year_flux_transpose.shape[2], block_year_flux_transpose.shape[3])
                    np.save(output_block_path + '/' + block + '_spatialtime.npy', flux_array)
                gc.collect()
                print('The ' + product + ' have done')
            print('The ' + block + ' have done')

        # product_list = ['MCD43A4'] # ,'MCD15A3H','DAILY_AGGR'
        # for product in product_list:
        #     # 进入每个州下面的blocks目录
        #     blocks_list = os.listdir(state_path)
        #     for k in range(0, len(blocks_list)):
        #         block = blocks_list[k]
        #         blocks_path = os.path.join(state_path, block)
        #
        #         output_block_path = os.path.join(output_state_path, block)
        #         if not os.path.isdir(output_block_path):
        #             os.makedirs(output_block_path)
        #         # 获取ref文件
        #         ref_files = (file for file in os.listdir(blocks_path) if
        #                      file.startswith('MCD43A4') and file.endswith('.tif'))
        #         # 再从这些files里获取日期list
        #         date_list = [file.split('_')[1].split('.')[0] for file in ref_files]
        #         block_image_year_list = []
        #         block_spatialtime_year_list = []
        #         for i in range(0, len(date_list)):
        #             file_name = product + '_' + date_list[i] + '.tif'
        #             file_path = os.path.join(blocks_path, file_name)
        #             if product == 'MCD43A4':
        #                 # 获取文件名
        #                 if not os.path.isfile(file_path):
        #                     ref_datasets = np.full((32, 32, 14, 3, 3), -9999, dtype='float64')
        #                     lon_lat_array = np.full((32, 32, 2), -9999, dtype='float64')
        #                 else:
        #                     # 再按3*3滑动窗口大小读取这个数据
        #                     ref_datasets, lon_lat_array = ref_read_multiband(file_path)
        #                 # ref_datasets = np.array(ref_datasets, dtype=np.float)
        #                 ref_datasets = ref_datasets.astype(np.float64)
        #                 block_image_year_list.append(ref_datasets)
        #                 # # 再按3*3滑动窗口大小读取这个数据
        #                 year, doy = out_day_by_date(date_list[i])
        #                 # 将year和doy填充到lon_lat_array中  变成 32，32，4 （4分别代表这个像素的lon，lat,year,doy）
        #                 # 在lon_lat_array(32, 32, 2)中填充year和doy，变成 (32, 32, 4)
        #                 spatialtime_array = np.dstack((
        #                     lon_lat_array, np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), year),
        #                     np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), doy)))
        #                 # 添加每个像素的空间位置和时间信息
        #                 spatialtime_array = spatialtime_array.astype(np.float64)
        #                 # spatialtime_array = np.array(spatialtime_array, dtype=np.float)
        #                 block_spatialtime_year_list.append(spatialtime_array)
        #             elif product == 'MCD15A3H':
        #                 if not os.path.isfile(file_path):
        #                     lai_datasets = np.full((32, 32, 4, 3, 3), -9999, dtype='float64')
        #                 else:
        #                     lai_datasets = read_multiband(file_path)
        #                 lai_datasets = lai_datasets.astype(np.float64)
        #                 block_image_year_list.append(lai_datasets)
        #             elif product == 'DAILY_AGGR':
        #                 if not os.path.isfile(file_path):
        #                     lai_datasets = np.full((32, 32, 6, 3, 3), -9999, dtype='float64')
        #                 else:
        #                     lai_datasets = read_multiband(file_path)
        #                 lai_datasets = lai_datasets.astype(np.float64)
        #                 block_image_year_list.append(lai_datasets)
        #         images_array = transpose_reshape(block_image_year_list)
        #         # 将这几个array输出保存为.npy格式
        #         np.save(output_block_path + '/' + block + '_' + product + '.npy', images_array)
        #         if product == 'MCD43A4':
        #             spatialtime_array = np.asarray(block_spatialtime_year_list)
        #             # 将数组维度变为（32，32，365，3，3）
        #             block_year_flux_transpose = np.transpose(spatialtime_array, (1, 2, 0, 3))
        #             # reshape
        #             flux_array = block_year_flux_transpose.reshape(
        #                 block_year_flux_transpose.shape[0] * block_year_flux_transpose.shape[1],
        #                 block_year_flux_transpose.shape[2], block_year_flux_transpose.shape[3])
        #             np.save(output_block_path + '/' + block + '_spatialtime.npy', flux_array)
        #         gc.collect()
        #         print('The ' + block + ' have done')
        #     print('The ' + product + ' have done')
        # block_ref_year_list = []
        # block_lai_year_list = []
        # block_mete_year_list = []
        # block_spatialtime_year_list = []
        # # 获取ref文件 以MCD43A4开头，日期结尾；e.g. MCD43A4_2017-01-01
        # # 依次遍历以MCD43A4开头的ref文件，同时得到ref的日期
        # # ref_files_list = [file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif')]
        # # for k in range(0, len(ref_files_list)):
        # # 使用生成器表达式
        # ref_files = (file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif'))
        # for ref_filename in ref_files:
        #     # 得到ref文件的路径
        #     ref_file_name = ref_filename
        #     ref_doy_path = os.path.join(blocks_path, ref_file_name)
        #     # 获取文件名
        #     # 再按3*3滑动窗口大小读取这个数据
        #     ref_datasets, lon_lat_array = ref_read_multiband(ref_doy_path)
        #     block_ref_year_list.append(ref_datasets)
        #     # 提取日期信息
        #     ref_file_str = ref_file_name.replace('.tif', '')  # 移除文件扩展名
        #     ref_date = ref_file_str.split('_')[1]  # 以'_'分割文件名
        #     # 返回pixel_index,lat,lon,year,doy
        #     # 返回影像的每个像素的位置索引和中心经纬度坐标
        #     year,doy = out_day_by_date(ref_date)
        #     # 将year和doy填充到lon_lat_array中  变成 32，32，4 （4分别代表这个像素的lon，lat,year,doy）
        #     # 在lon_lat_array(32, 32, 2)中填充year和doy，变成 (32, 32, 4)
        #     spatialtime_array = np.dstack((lon_lat_array, np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), year), np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), doy)))
        #     # 添加每个像素的空间位置和时间信息
        #     block_spatialtime_year_list.append(spatialtime_array)
        #
        #     ## 加载lai数据
        #     lai_filename = 'MCD15A3H_'+ref_date+'.tif'
        #     lai_doy_path = os.path.join(blocks_path, lai_filename)
        #     if os.path.exists(lai_doy_path):
        #         lai_datasets = read_multiband(lai_doy_path)
        #         block_lai_year_list.append(lai_datasets)
        #     else:
        #         lai_datasets = np.full((32, 32, 4, 3, 3), -9999)
        #         block_lai_year_list.append(lai_datasets)
        #
        #     # 加载meteorology数据
        #     mete_filename = 'DAILY_AGGR_' + ref_date + '.tif'
        #     mete_doy_path = os.path.join(blocks_path, mete_filename)
        #     if os.path.exists(mete_doy_path):
        #         mete_datasets = read_multiband(mete_doy_path)
        #         block_mete_year_list.append(mete_datasets)
        #     else:
        #         mete_datasets = np.full((32, 32, 6, 3, 3), -9999)
        #         block_mete_year_list.append(mete_datasets)
        #
        # # 将list转为数组,维度变换 n,366,b,3,3
        # ref_array = transpose_reshape(block_ref_year_list)
        # lai_array = transpose_reshape(block_lai_year_list)
        # mete_array = transpose_reshape(block_mete_year_list)
        # spatialtime_array = np.array(block_spatialtime_year_list)
        # # 将数组维度变为（32，32，365，3，3）
        # block_year_flux_transpose = np.transpose(spatialtime_array, (1, 2, 0, 3))
        # # reshape
        # flux_array = block_year_flux_transpose.reshape(block_year_flux_transpose.shape[0]*block_year_flux_transpose.shape[1],
        #                                                           block_year_flux_transpose.shape[2],block_year_flux_transpose.shape[3])
        #
        # # 将这几个array输出保存为.npy格式
        # np.save(output_block_path+'/'+block+'_ref.npy', ref_array)
        # np.save(output_block_path+'/'+block+'_lai.npy', lai_array)
        # np.save(output_block_path+'/'+block+'_mete.npy', mete_array)
        # np.save(output_block_path+'/'+block+'_spatialtime.npy', flux_array)
        #
        # # 释放内存
        # del ref_array, lai_array, mete_array, spatialtime_array, block_year_flux_transpose, flux_array
        # gc.collect()
        # print('The'+block+'have done')

