
# 提取3*3窗口的反射率、lai
import os
import glob

import pandas as pd
import pyproj
import rasterio
from osgeo import gdal
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from datetime import datetime
import h5py
from datetime import datetime, timedelta

def resubset_values(point,img_tif):
    print('hhh')
    # 计算point在tif文件中的行列号位置坐标

    # 打开影像文件
    # ## ################################
    # gdal方式打开tif文件
    ds = gdal.Open(img_tif)
    # 读取中心像元的值
    # 获取tif文件的宽度和高度
    width = ds.RasterXSize
    height = ds.RasterYSize
    # 计算中心像元的坐标
    x_center = width // 2
    y_center = height // 2
    center_pixel_value = ds.ReadAsArray(x_center, y_center, 1, 1)
    # 以中心像元为中心，读取3x3窗口像元
    single_band_data = ds.ReadAsArray(x_center - 1, y_center - 1, 3, 3)
    print(single_band_data)
    # 获取影像的投影信息
    im_proj = ds.GetProjection()
    # 获取影像的仿射变换参数
    im_geotrans = ds.GetGeoTransform()
    ## ################################
    # rasterio 方式
    # # 创建投影坐标系的转换器
    # proj_in = pyproj.CRS.from_epsg(4326)  # 输入坐标系为WGS84经纬度坐标系
    # proj_out = pyproj.CRS.from_epsg(3857)  # 输出坐标系为EPSG:3857投影坐标系
    # # # 将经纬度坐标转换为投影坐标
    # # 定义转换器
    # transformer = pyproj.Transformer.from_crs(proj_in, proj_out, always_xy=True)

    # # 进行坐标转换
    # x, y = transformer.transform(point.x, point.y)
    # # 在此处编写代码以获取 single_band_data, im_proj, im_geotrans
    # single_band_data = None  # 假设默认值为 None
    # im_proj = None
    # im_geotrans = None
    # with rasterio.open(img_tif) as src:
    #     print(src.crs)
    #     # 判断 `img_tif` 是否为 `None`
    #     if src.crs is None:
    #         print('src is None')
    #     else:
    #         # 将点地理坐标转换为像素坐标
    #         point_coords = src.index(x, y)
    #         # 获取行列号（像素坐标）
    #         row, col = int(point_coords[0]), int(point_coords[1])
    #         print(row,col)
    #         print(src.height, src.width)
    #
    #         # 定义影像块的大小
    #         block_size = 3  # 影像块的大小为50x50
    #
    #         # 计算影像块的左上角像素坐标和宽高
    #         left = col - block_size // 2
    #         top = row - block_size // 2
    #         width = block_size
    #         height = block_size
    #
    #         # 从原始影像中读取指定大小的影像块
    #         window = Window(left, top, width, height)
    #         single_band_data = src.read(1, window=window)  # 读取第一个波段的像素值
    #         print(single_band_data.shape)
    #         # 在这里判断window_data是不是空的，如果是需要跳过进行处理
    #         # 获取投影信息和仿射变换信息
    #         im_proj = src.crs.to_string()  # 获取投影信息
    #         # im_geotrans = src.transform.to_gdal()  # 获取的是原影像的仿射变换信息
    #         # 获取从窗口提取的影像块data的仿射变换信息
    #         window_transform = src.window_transform(window)
    #
    #         # 将windows提取的影像块的仿射变换信息转换为GDAL格式
    #         im_geotrans = window_transform.to_gdal()

    return single_band_data,im_proj,im_geotrans

def read_multiband(imagepath,point,bandlist):
    # 查找匹配的文件路径
    re_datasets = []
    file_path_name = imagepath.split('\\')  # 根据 '\\' 进行分割
    file_name = file_path_name[-1]
    for keyword in bandlist:
        file_finalname = file_name +'_' + keyword +'.tif'
        image_filepath = os.path.join(imagepath,file_finalname)
        # 将dataset提取为50*50大小
        dataset, im_proj, im_geotrans = resubset_values(point, image_filepath)
        masked_dataset = np.copy(dataset)
        # 将不等于-9999的转换为地表反射率
        ## 需要进行判断，如果是地表反射率则乘以比例因子，如果不是则不需要乘
        if 'Reflectance' in keyword:
            reflect_dataset = np.where(masked_dataset != -9999, masked_dataset * 0.0001, masked_dataset)
        elif keyword == 'Fpar':
            reflect_dataset = np.where(masked_dataset != -9999, masked_dataset * 0.01, masked_dataset)
        elif keyword == 'Lai':
            reflect_dataset = np.where(masked_dataset != -9999, masked_dataset * 0.1, masked_dataset)
        else:
            reflect_dataset = masked_dataset
        re_datasets.append(reflect_dataset)
    re_datasets = np.array(re_datasets)
    return re_datasets,im_proj,im_geotrans

# 将日期转为数字
def out_day_by_date(Date):
    '''
    根据输入的日期计算该日期是在当年的第几天
    '''
    print(Date)
    # 将字符串转为日期格式
    Date = datetime.strptime(Date, '%Y_%m_%d').date()
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
    return year,sum,Date

# 将每个站点&每一年的数据拼接起来组成一个输入数据
def concatenate_and_pad_time_series(ref_list):
    new_list = []
    max_shape = max(max(arr.shape[0] for arr in sublist) for sublist in ref_list)
    for sublist in ref_list:
        for arr in sublist:
            if arr.shape[0] < max_shape:
                padding_shape = (max_shape - arr.shape[0],) + arr.shape[1:]
                padding = np.full(padding_shape, -9999)
                new_arr = np.concatenate((arr, padding), axis=0)
                new_list.append(new_arr)
            else:
                new_list.append(arr)
    combined_array = np.array(new_list)
    return combined_array

if __name__ == '__main__':
    # 下面的功能主要是对fluxnet变量进行聚合，从半小时聚合到天和月，然后是提取一定窗口大小的影像50*50
    # 最终输出要输入模型的x_image_array和y_flux_array
    # shp_path = "E:/Carbon_flux/data/originate_data/site_shp/Sites_HH_GE_shar_6.shp"
    shp_path = "E:/Carbon_flux/data/originate_data/site_shp/sites_reduced_HH_GE_shar_270.shp"
    # 读取shp文件
    points = gpd.read_file(shp_path)
    # 总的data array，后面每添加一个x_array就添加一个time_array和site_array
    ref_list = []
    lai_list = []
    flux_list = []
    mete_list = []
    # 利用窗口重采样之后 影像数据存储的位置
    re_image_root_path = 'E:/Carbon_flux/data/sites_image3_resample'
    # 聚合后的Y变量存储的路径
    flux_df_root_path = 'E:/Carbon_flux/data/originate_data/GE_AmeriFlux_data/GE_AmeriFlux_data_DD_MM'
    # 反射率和lai影像存储的根路径
    site_image_path = 'E:/Carbon_flux/data/modis'
    # 分别进入反射率和lai影像子文件夹
    ref_image_path = os.path.join(site_image_path, 'sites_image_modis_ref_MCD43A4')
    lai_image_path = os.path.join(site_image_path, 'sites_image_modis_fpar_lai_mcd15a3h')
    mete_image_path = os.path.join(site_image_path, 'Sites_image_ERA5_LAND_DAILY_AGGR')
    # 统一路径中的斜杠
    ref_image_path = ref_image_path.replace('\\', '/')
    lai_image_path = lai_image_path.replace('\\', '/')
    mete_image_path = mete_image_path.replace('\\', '/')

    # site目录
    ref_sites_list = os.listdir(ref_image_path)
    lai_sites_list = os.listdir(lai_image_path)
    mete_sites_list = os.listdir(mete_image_path)
    # 同时进入两个子文件夹下的site目录
    # for i in range(0,len(ref_sites_list)):
    for i in range(0,len(ref_sites_list)): # len(ref_sites_list)
        site = ref_sites_list[i]
        point = points.geometry[i]
        lon = point.x
        lat = point.y
        # 进入反射率站点目录下的各个年份目录
        ref_sites_year_path = os.path.join(ref_image_path, ref_sites_list[i])
        # 进入ref_site_year list
        ref_sites_year_list = os.listdir(ref_sites_year_path)

        # 进入lai站点目录下的各个年份目录
        lai_sites_year_path = os.path.join(lai_image_path, site)
        # 进入ref_site_year list
        lai_sites_year_list = os.listdir(lai_sites_year_path)

        # 进入mete站点目录下的各个年份目录
        mete_sites_year_path = os.path.join(mete_image_path, site)
        # 进入ref_site_year list
        mete_sites_year_list = os.listdir(mete_sites_year_path)

        ref_site_lists = []
        lai_site_lists = []
        flux_site_lists = []
        mete_site_lists = []
        #进入ref_site_year_list lai_site_year_list  len(ref_sites_year_list)
        for m in range(0,len(ref_sites_year_list)):
        # for m,n in zip(range(len(ref_sites_year_list)), range(len(lai_sites_year_list))):
            ref_site_year = ref_sites_year_list[m]
            lai_site_year = lai_sites_year_list[m]
            mate_site_year = mete_sites_year_list[m]
            ref_site_year_path = os.path.join(ref_sites_year_path, ref_sites_year_list[m])
            lai_site_year_path = os.path.join(lai_sites_year_path, lai_sites_year_list[m])
            mete_site_year_path = os.path.join(mete_sites_year_path, mete_sites_year_list[m])
            # 进入ref_site_year_path目录 读取这一年每一天的数据
            # 起始日期及结束日期
            start_date = datetime(int(ref_site_year), 1, 1)
            end_date = datetime(int(ref_site_year) + 1, 1, 1)
            # 生成日期列表
            date_list = [(start_date + timedelta(days=i)).strftime('%Y_%m_%d') for i in range((end_date - start_date).days)]
            # ref_site_year_doy_list = os.listdir(ref_site_year_path)
            # ref_site_year_doy_list = list(set(ref_site_year_doy_list))
            # 将这一年的数据加入一个list
            ref_year_list = []
            lai_year_list = []
            mete_year_list = []
            time_year_list = []
            site_year_list = []
            for k in range(0,len(date_list)):
                data_date = date_list[k]
                print(data_date )
                ref_site_year_doy_path = os.path.join(ref_site_year_path, data_date+'_'+'MCD43A4')
                # ref_site_year_doy_path = os.path.join(ref_site_year_path, ref_site_year_doy_list[k])
                # 依次读取ref_site_year_doy_path文件夹下的反射率波段和对应的QA波段，前7个波段是反射率，紧接着是每个波段对应的qa波段
                ref_bandlist = ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3',
                                'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7',
                                'BRDF_Albedo_Band_Mandatory_Quality_Band1','BRDF_Albedo_Band_Mandatory_Quality_Band2','BRDF_Albedo_Band_Mandatory_Quality_Band3',
                                'BRDF_Albedo_Band_Mandatory_Quality_Band4','BRDF_Albedo_Band_Mandatory_Quality_Band5','BRDF_Albedo_Band_Mandatory_Quality_Band6','BRDF_Albedo_Band_Mandatory_Quality_Band7']
                if not os.path.isdir(ref_site_year_doy_path):
                    ref_datasets= np.full((14, 3, 3), -9999)
                    # print(lai_site_year_doy_path)
                    ref_year_list.append(ref_datasets)
                else:
                    ref_datasets, ref_proj, ref_geotrans = read_multiband(ref_site_year_doy_path,point,ref_bandlist)
                    print(ref_datasets)
                    ref_year_list.append(ref_datasets)
                # ref_site_year_doy_date = ref_site_year_doy_list[k][0:10]
                # print(ref_site_year_doy_date)
                lai_site_year_doy_path = os.path.join(lai_site_year_path, data_date +'_MCD15A3H')
                lai_bandlist = ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
                if not os.path.isdir(lai_site_year_doy_path):
                    lai_datasets = np.full((4, 3, 3), -9999)
                    lai_year_list.append(lai_datasets)
                else:
                    lai_datasets, lai_proj, lai_geotrans = read_multiband(lai_site_year_doy_path,point,lai_bandlist)
                    lai_year_list.append(lai_datasets)
                # mete影像
                mete_site_year_doy_date = data_date.replace('_', '')
                mete_site_year_doy_path = os.path.join(mete_site_year_path, mete_site_year_doy_date + '_ERA5_LAND_DAILY_AGGR')
                mete_bandlist = ['surface_solar_radiation_downwards_sum','temperature_2m','volumetric_soil_water_layer_1',
                                 'volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']
                if not os.path.isdir(mete_site_year_doy_path):
                    mete_datasets = np.full((6, 3, 3), -9999)
                    mete_year_list.append(mete_datasets)
                else:
                    mete_datasets, mete_proj, mete_geotrans = read_multiband(mete_site_year_doy_path,point,mete_bandlist)
                    mete_year_list.append(mete_datasets)
                # 返回影像的年份和doy
                year, doy, Date = out_day_by_date(data_date)
                time_year_list.append((year, doy))
                # 返回影像对应的site序列号、经度和维度
                site_year_list.append((site, lon, lat))
                print(site, year, doy, Date)
            print(ref_sites_year_list[m])
            # print(site, ref_site_year_doy_list[k])
            # 将一年的list转为array
            ref_year_array = np.array(ref_year_list)
            lai_year_array = np.array(lai_year_list)
            mete_year_array = np.array(mete_year_list)
            ref_site_lists.append(ref_year_array)
            lai_site_lists.append(lai_year_array)
            mete_site_lists.append(mete_year_array)
            # 根据year doy匹配flux通量数据
            flux_sites_path = os.path.join(site_image_path, 'GE_AmeriFlux_data_DD_MM')
            flux_sites_year_list = os.listdir(flux_sites_path)
            site_array = np.asarray(site_year_list)
            # print(site_array[1, :])
            time_array = np.asarray(time_year_list)
            # print(time_array[1, :])
            time_df = pd.DataFrame(time_array, columns=['Year', 'DoY'])
            site_df = pd.DataFrame(site_array[:, :], columns=['Site_Id', 'longitude', 'latitude'])
            # 实际的站点-影像-doy
            Sites_time_df = pd.concat([site_df, time_df], axis=1)
            flux_df_path_name = site + '_DD_MM_' + str(year)+'.csv'
            # 遍历指定路径及其子文件夹下的所有文件
            # 打开对应年份的数据
            flux_sites_year_path = os.path.join(flux_df_root_path, site)
            flux_sites_year_doy_path =  os.path.join(flux_sites_year_path, flux_df_path_name)
            # 进行判断
            if os.path.exists(flux_sites_year_doy_path):
                # 打开文件并读取数据
                Y_df = pd.read_csv(flux_sites_year_doy_path)
                print(Y_df.head())
                # 将实际的X_df(Sites_time_df)与Y_df 根据'Site_Id', 'Year', 'DoY' 进行匹配
                Flux_image_Y_df = pd.merge(Sites_time_df, Y_df, on=['Site_Id', 'Year', 'DoY'], how='left')
                print(len(Flux_image_Y_df))
                # 将没有匹配值的设置为-9999
                Flux_image_Y_df = Flux_image_Y_df.fillna(-9999)
                print(Flux_image_Y_df.head())
                flux_site_lists.append(np.array(Flux_image_Y_df))
                print('hhh')
            else:
                num_rows = Sites_time_df.shape[0]  # 获取行数
                num_columns = 23  # 获取列数
                Flux_image_Y_df = pd.DataFrame(-9999, index=range(num_rows), columns=range(num_columns))
                flux_site_lists.append(np.array(Flux_image_Y_df))
                print('the flux data of this year is not exist')
            print(flux_sites_year_doy_path)
            # 将 Flux_image_Y_df存储下来
            # Flux_image_Y_df.to_csv('E:/Carbon_flux/data/modis/Flux_image_Y_df_example.csv')
            # 网格级的气象数据
        print(ref_site_lists)
        print(lai_site_lists)
        print(flux_site_lists)
        print(mete_site_lists)
        ref_list.append(ref_site_lists)
        lai_list.append(lai_site_lists)
        flux_list.append(flux_site_lists)
        mete_list.append(mete_site_lists)
    # 所有站点的array
    print('lllll')
    print(ref_list)
    # 根据list 的最大time series进行padding
    ref_array = concatenate_and_pad_time_series(ref_list)
    lai_array = concatenate_and_pad_time_series(lai_list)
    flux_array = concatenate_and_pad_time_series(flux_list)
    mete_array = concatenate_and_pad_time_series(mete_list)
    ref_array[:,:,7,1,1]
    print(ref_array.shape)
    print(lai_array.shape)
    print(flux_array.shape)
    print(mete_array.shape)
    # 假设 mete_array 是您的数组变量
    # # 遍历每个波段
    # for i in range(6):
    #     band_values = mete_array[:, :, i, :, :]
    #     # 过滤掉非有限的值
    #     finite_values = band_values[np.isfinite(band_values)]
    #     # # 创建掩码来替换多个条件
    #     # mask = (band_values == -9999) | (band_values == 55537) | (band_values == -10000)
    #     #
    #     # # 将符合条件的元素替换为NaN
    #     # band_values[mask] = np.nan
    #     # 计算最大最小值
    #     max_value = np.nanmax(finite_values)
    #     min_value = np.nanmin(finite_values)
    #     print(f"波段 {i + 1} 除去-9999以外的最大值为: {max_value}, 最小值为: {min_value}")
    inputdata_path = 'E:/Paper code/code/Carbon_flux_estimation/data/sites_inputdata/'
    np.save(inputdata_path + 'x_ref_modis_new.npy', ref_array)
    np.save(inputdata_path + 'x_lai_modis_new.npy', lai_array)
    np.save(inputdata_path + 'x_mete_era5_new.npy', mete_array)
    np.save(inputdata_path + 'y_flux_meteorological_new.npy', flux_array)
