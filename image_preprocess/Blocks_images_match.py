#

import os
import numpy as np
from osgeo import gdal
import rasterio
from numpy.lib.stride_tricks import as_strided
from datetime import datetime
from rasterio.windows import Window

def ref_read_multiband(ref_doy_path):
    # 读取并打开文件
    with rasterio.open(ref_doy_path) as src:
        # 读取所有波段到一个numpy数组中
        data = src.read()
        # 获取波段数和数组的维度
        num_bands, height, width = data.shape
        # 定义窗口大小
        window_size = 3
        # 计算输出数组的形状
        output_shape = (height, width, num_bands, window_size, window_size)
        # 重塑数组以创建滑动窗口视图
        sub_arrays = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=data.itemsize * np.array(
            [width * num_bands, num_bands, width * num_bands, num_bands, 1]))
        lon_list, lat_list = [], []
        for i in range(height):
            for j in range(width):
                lon, lat = src.xy(i, j)
                lon_list.append(lon)
                lat_list.append(lat)

        lon_lat_array = np.array(list(zip(lon_list, lat_list))).reshape((height, width, 2))

        print(sub_arrays.shape)
        return sub_arrays, lon_lat_array


def read_multiband(ref_doy_path):
    # 读取并打开文件
    # Open the multi-band TIFF file
    with rasterio.open(ref_doy_path) as src:
        # Read all bands into a single numpy array
        data = src.read()

    # Get the number of bands and the dimensions of the array
    num_bands, height, width = data.shape

    # Define the window size
    window_size = 3

    # Calculate the shape of the output array
    output_shape = (height, width, num_bands, window_size, window_size)

    # Reshape the array to create the sliding window view
    sub_arrays = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=data.itemsize * np.array(
        [width * num_bands, num_bands, width * num_bands, num_bands, 1]))
    # The sub_arrays variable now contains the data arranged in the required shape
    print(sub_arrays.shape)
    return sub_arrays

# 将日期转为数字
def out_day_by_date(Date):
    '''
    根据输入的日期计算该日期是在当年的第几天
    '''
    print(Date)
    # 将字符串转为日期格式
    Date = datetime.strptime(Date, '%Y-%m-%d').date()
    # date = Date.apply(lambda x: datetime.strptime(x,'%Y%m%d'))
    print(Date)
    year = Date.year
    month = Date.month
    day = Date.day
    # year = date[0:4]
    # month = date[4:6]
    # day = int(date[6:8])
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

# 将list转为数组，并变换维度
def transpose_reshape(block_ref_year_list):
    block_year_array = np.array(block_ref_year_list)
    # 将数组维度变为（32，32，365，14,3，3）
    block_year_array_transpose = np.transpose(block_year_array, (1, 2, 0, 3, 4,5))
    # reshape 将第0维和第1维乘起来，其余维度保持不变
    block_year_array_reshape = block_year_array_transpose.reshape(block_year_array_transpose.shape[0]*block_year_array_transpose.shape[1],
                                                                  block_year_array_transpose.shape[2],block_year_array_transpose.shape[3],
                                                                 block_year_array_transpose.shape[4],block_year_array_transpose.shape[5])
    return block_year_array_reshape

if __name__ == '__main__':
    # 所有region 存储的根目录 每个国家的文件夹
    region_base_dir = 'E:/Carbon_flux/data/regoin_image'

    output_path = 'E:/Carbon_flux/data/regoin_image/region_images_test_blocks_match/'
    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_base_dir)
    nation_path = os.path.join(region_base_dir, nation_list[4])

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    state_path = os.path.join(nation_path,state_list[0])

    output_state_path = os.path.join(output_path,state_list[0])
    if not os.path.exists(output_state_path):
        os.makedirs(output_state_path)

    # 进入每个州下面的blocks目录
    for block in os.listdir(state_path):
        output_block_path = os.path.join(output_state_path, block)
        if not os.path.exists(output_block_path):
            os.makedirs(output_block_path)

        blocks_path = os.path.join(state_path,block)
        block_ref_year_list = []
        block_lai_year_list = []
        block_mete_year_list = []
        block_spatialtime_year_list = []
        # 获取ref文件 以MCD43A4开头，日期结尾；e.g. MCD43A4_2017-01-01
        # 依次遍历以MCD43A4开头的ref文件，同时得到ref的日期
        # ref_files_list = [file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif')]
        # for k in range(0, len(ref_files_list)):
        # 使用生成器表达式
        ref_files = (file for file in os.listdir(blocks_path) if file.startswith('MCD43A4') and file.endswith('.tif'))
        for ref_filename in ref_files:
            # 得到ref文件的路径
            ref_file_name = ref_filename
            ref_doy_path = os.path.join(blocks_path, ref_file_name)
            # 获取文件名
            # 再按3*3滑动窗口大小读取这个数据
            ref_datasets, lon_lat_array = ref_read_multiband(ref_doy_path)
            ref_datasets[:,0,7,0,0]
            block_ref_year_list.append(ref_datasets)
            # 提取日期信息
            ref_file_str = ref_file_name.replace('.tif', '')  # 移除文件扩展名
            ref_date = ref_file_str.split('_')[1]  # 以'_'分割文件名
            # 返回pixel_index,lat,lon,year,doy
            # 返回影像的每个像素的位置索引和中心经纬度坐标
            year,doy = out_day_by_date(ref_date)
            # 将year和doy填充到lon_lat_array中  变成 32，32，4 （4分别代表这个像素的lon，lat,year,doy）
            # 在lon_lat_array(32, 32, 2)中填充year和doy，变成 (32, 32, 4)
            spatialtime_array = np.dstack((lon_lat_array, np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), year), np.full((lon_lat_array.shape[0], lon_lat_array.shape[1], 1), doy)))
            # 添加每个像素的空间位置和时间信息
            block_spatialtime_year_list.append(spatialtime_array)

            ## 加载lai数据
            lai_filename = 'MCD15A3H_'+ref_date+'.tif'
            lai_doy_path = os.path.join(blocks_path, lai_filename)
            if os.path.exists(lai_doy_path):
                lai_datasets = read_multiband(lai_doy_path)
                block_lai_year_list.append(lai_datasets)
            else:
                lai_datasets = np.full((32, 32, 4, 3, 3), -9999)
                block_lai_year_list.append(lai_datasets)

            # 加载meteorology数据
            mete_filename = 'DAILY_AGGR_' + ref_date + '.tif'
            mete_doy_path = os.path.join(blocks_path, mete_filename)
            if os.path.exists(mete_doy_path):
                mete_datasets = read_multiband(mete_doy_path)
                block_mete_year_list.append(mete_datasets)
            else:
                mete_datasets = np.full((32, 32, 6, 3, 3), -9999)
                block_mete_year_list.append(mete_datasets)

        # 将list转为数组,维度变换 n,366,b,3,3
        ref_array = transpose_reshape(block_ref_year_list)
        lai_array = transpose_reshape(block_lai_year_list)
        mete_array = transpose_reshape(block_mete_year_list)
        spatialtime_array = np.array(block_spatialtime_year_list)
        # 将数组维度变为（32，32，365，3，3）
        block_year_flux_transpose = np.transpose(spatialtime_array, (1, 2, 0, 3))
        # reshape
        flux_array = block_year_flux_transpose.reshape(block_year_flux_transpose.shape[0]*block_year_flux_transpose.shape[1],
                                                                  block_year_flux_transpose.shape[2],block_year_flux_transpose.shape[3])

        # 将这几个array输出保存为.npy格式
        np.save(output_block_path+'/'+block+'_ref.npy', ref_array)
        np.save(output_block_path+'/'+block+'_lai.npy', lai_array)
        np.save(output_block_path+'/'+block+'_mete.npy', mete_array)
        np.save(output_block_path+'/'+block+'_spatialtime.npy', flux_array)

        # 释放内存
        del ref_array, lai_array, mete_array, spatialtime_array, block_year_flux_transpose, flux_array
        print('The'+block+'have done')

