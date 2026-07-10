# Initialize earth engine
import ee
import rasterio as rasterio

service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/GEE_json/biomass-estimates-d58a1f0e77e5.json")
ee.Initialize(credentials)

import pandas as pd
from gee_subset import gee_subset
import utility
import numpy as np
from osgeo import gdal
from pyproj import Transformer
from pyproj import CRS
import time as t
from math import cos, radians
from pyproj import Proj, transform
from osgeo import gdal
from rasterio.transform import from_origin

# # read the site.csv
# site_df = pd.read_csv("E:/2_biomass/data/sites_reduced_HH_GE_share.csv")

import pandas as pd
from osgeo import gdal, osr
import os
import datetime

output_file = 'output.tif'

def get_geotransform(df):
    # 将经纬度换成WGS 1984 Web 3857投影坐标系坐标
    # 定义坐标转换器（从经纬度转换为3857坐标）
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    # 转换经纬度坐标为3857投影坐标
    df['longitude_prj'], df['latitude_prj'] = transformer.transform(df['longitude'].values, df['latitude'].values)
    print(df.head())
    # 设置影像宽、高和波段的数量
    width = len(df['longitude'].unique())
    height = len(df['latitude'].unique())
    # num_bands = len(df.columns) - 3  # 减去经度、纬度和日期三列
    # 添加地理变换信息
    xmin, ymin, xmax, ymax = df['longitude_prj'].min(), df['latitude_prj'].min(), df['longitude_prj'].max(), df['latitude_prj'].max()
    # 计算仿射变换参数  针对modis系列数据
    pixel_width = (xmax - xmin) / (width - 1)
    pixel_height = (ymax - ymin) / (height - 1)
    xxmin = df['longitude_prj'].min() - pixel_width/2
    yymax = df['latitude_prj'].max() + pixel_height/2
    geotransform = (xxmin, pixel_width, 0, yymax, 0, -pixel_height)
    return geotransform

# 针对气象数据
# def get_geotransform(df):
#     # 将经纬度换成WGS 1984 Web 3857投影坐标系坐标
#     # 定义坐标转换器（从经纬度转换为3857坐标）
#     transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
#     # 转换经纬度坐标为3857投影坐标
#     df['longitude_prj'], df['latitude_prj'] = transformer.transform(df['longitude'].values, df['latitude'].values)
#     print(df.head())
#     # 设置重采样的分辨率
#     resolution = 500
#     half_resolution = resolution / 2
#     xmin = df['longitude_prj'] - (half_resolution + resolution)
#     ymax = df['latitude_prj'] + (half_resolution + resolution)
#
#     pixel_width = resolution
#     pixel_height = resolution  # 由于像素高度通常为负值
#     geotransform = (xmin, pixel_width, 0, ymax, 0, -pixel_height)
#     return geotransform


# def get_geotransform(df):
#     # 将经纬度换成WGS 1984 Web 3857投影坐标系坐标
#     transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
#     # 转换经纬度坐标为3857投影坐标
#     df['longitude_prj'], df['latitude_prj'] = transformer.transform(df['longitude'].values, df['latitude'].values)
#     print(df.head())
#     # 判断南北半球情况
#     is_northern_hemisphere = df['latitude_prj'].max() > 0
#     width = len(df['longitude_prj'].unique())
#     height = len(df['latitude_prj'].unique())
#     # 添加地理变换信息
#     min_longitude, min_latitude = df['longitude_prj'].min(), df['latitude_prj'].min()
#     max_longitude, max_latitude = df['longitude_prj'].max(), df['latitude_prj'].max()
#     # 根据南北半球情况进行计算
#     if is_northern_hemisphere:
#         pixel_width = (max_longitude - min_longitude) / (width - 1)
#         pixel_height = -(max_latitude - min_latitude) / (height - 1)
#     else:
#         pixel_width = (max_longitude - min_longitude) / (width - 1)
#         pixel_height = -(min_latitude - max_latitude) / (height - 1)
#     geotransform = (min_longitude, pixel_width, 0, max_latitude, 0, pixel_height)
#     return geotransform

def calculate_utm_zone(longitude):
    zone = int((longitude + 180) / 6) + 1
    if longitude < 0:
        zone *= -1
    return zone

def convert_meter_to_latlon(center_lon,meter):
    zone = str(abs(int((center_lon + 180) / 6)) + 1)  # 计算UTM区号
    xy_proj = Proj(proj='utm', zone=zone, south=True, datum='WGS84')  # 南半球
    # 假设转换为北半球，东经0度的经纬度
    lat, lon = xy_proj(0, meter, inverse=True)
    return lat, lon


def get_product_band(year):
    # year = str(year)
    # 将起止日期str转为日期类型
    start_date = str(year) + '-01-01'
    # start_date = datetime.date(*map(int, start_date.split('-')))
    end_date = str(year+1) + '-01-01'

    # end_date = datetime.date(*map(int, end_date.split('-')))
    # # 下载影像格式的气象数据 根据year判断产品类型和band
    # if year > 1999 and year < 2022:
    #     product = ['ECMWF/ERA5_LAND/DAILY_AGGR']
    #     band_list = ['surface_solar_radiation_downwards_sum','temperature_2m','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']
    #     # band_list = ['surface_solar_radiation_downwards_sum']
    # else:
    #     print('the data is not available')
    # if year > 1999 and year < 2022:
    #     product = ['MODIS/061/MCD43A4']
    #     band_list = ['BRDF_Albedo_Band_Mandatory_Quality_Band1','BRDF_Albedo_Band_Mandatory_Quality_Band2','BRDF_Albedo_Band_Mandatory_Quality_Band3',
    #                  'BRDF_Albedo_Band_Mandatory_Quality_Band4','BRDF_Albedo_Band_Mandatory_Quality_Band5','BRDF_Albedo_Band_Mandatory_Quality_Band6',
    #                  'BRDF_Albedo_Band_Mandatory_Quality_Band7',
    #                  'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3',
    #                  'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7']
    # else:
    #     print('the data is not available')
    # 下载LAi数据
    if year > 2001 and year < 2023:
        product = ['MODIS/061/MCD15A3H']
        band_list = ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC','FparStdDev','LaiStdDev']
    else:
        print('the data is not available')
    # if year > 1999 and year < 2023:
    #     product = ['MODIS/061/MOD15A2H']
    #     band_list = ['Fpar_500m','Lai_500m','FparLai_QC','FparExtra_QC','FparStdDev_500m','LaiStdDev_500m']
    # else:
    #     print('the data is not available')
    return start_date,end_date,product,band_list


def meters_to_lat_lon_displacement(m, origin_lat):
    """
    Rough calculation from meters to lat,lon to calculate buffer sizes
    from https://gis.stackexchange.com/a/2964/96775
    """
    lat = m/111111
    lon = m/(111111 * cos(radians(origin_lat)))
    return lat, lon


if __name__ == '__main__':
    sites_df = pd.read_csv("E:/Carbon_flux/data/input_data/Sites_HH_GE_shar_270.csv")
    # df = pd.read_csv('C:/Users/wuth/Desktop/image_df.csv')
    # df = pd.read_csv("E:/Carbon_flux/data/input_data/MODIS_Sites_test.csv")
    # print(df.head(5))
    ## 创建影像存储的根目录
    sites_image_root_path = 'E:/Carbon_flux/data/modis/Sites_mcd15a3h'
    # sites_image_root_path = 'E:\Carbon_flux\data\modis\Sites_image_ERA5_LAND_DAILY_AGGR'
    nodata_list = []
    nodata_df = pd.DataFrame()
    # for m in range(158,len(sites_df)):
    for m in range(0,1):
        sites_image_path_name = sites_df['Site_Id'][m]
        sites_image_path = os.path.join(sites_image_root_path, sites_image_path_name)
        # 检查文件夹是否存在
        if not os.path.isdir(sites_image_path):
            # 创建文件夹
            os.makedirs(sites_image_path)
            print(f"已创建文件夹：{sites_image_path}")
        else:
            print(f"文件夹已存在：{sites_image_path}")
        # year_list = []
        for year in range((sites_df['StartYear'][m]), (sites_df['EndYear'][m] + 1)):
            site_image_year_name = str(year)
            site_image_year_path = os.path.join(sites_image_path, site_image_year_name)
            # 检查文件夹是否存在
            if not os.path.isdir(site_image_year_path):
                # 创建文件夹
                os.makedirs(site_image_year_path)
                print(f"已创建文件夹：{site_image_year_path}")
            else:
                print(f"文件夹已存在：{site_image_year_path}")

            # # 根据年份判断产品类型和band
            # if year > 1999 and year < 2022:
            #     year = year
            # else:
            #     continue
            start_date, end_date, product_list, band_list = get_product_band(year)
            print(start_date, end_date, product_list, band_list)
            product = product_list[0]
            band = band_list
            df = gee_subset.gee_subset(product=product,
                                       bands=band,
                                       start_date=start_date,
                                       end_date=end_date,
                                       latitude=sites_df.loc[m, 'Latitude'],
                                       longitude=sites_df.loc[m, 'Longitude'],
                                       scale=500,
                                       pad=1)
            # print(df.head())
            # df.to_csv("E:/Carbon_flux/data/modis/MODIS_Sites_ERA5_TEST.csv")
            if df is None :
                nodata_name = sites_image_path_name + '_' + site_image_year_name + '_' + product.replace("/","_")
                nodata_list.append(nodata_name)
                nodata_df.loc[m,'site'] = sites_image_path_name
                nodata_df.loc[m, 'None'] = nodata_name
            elif df.empty:
                emptydata_name = sites_image_path_name + '_' + site_image_year_name + '_' + product.replace("/","_")
                nodata_list.append(emptydata_name)
                nodata_df.loc[m, 'site'] = sites_image_path_name
                nodata_df.loc[m, 'empty'] = emptydata_name
            else:
                # 按照id进行升序
                df.sort_values(by='id', inplace=True, ascending=True)
                # 将date列解析为日期时间格式
                df['date'] = pd.to_datetime(df['date'])
                # 提取日期并将其格式化为 'YYYY_MM_DD'
                df['formatted_date'] = df['date'].dt.strftime('%Y_%m_%d')
                # 提取唯一的日期值并转换为列表
                date_list = df['formatted_date'].unique().tolist()
                # 提取唯一的id值至列表
                id_list = df['id'].unique().tolist()
                # df.to_csv('./data/UHo3.csv')
                # 将df中的None值用-9999来填充
                df.fillna(value=-9999, method=None, axis=None, inplace=True)
                # 在这里应该就按照id 日期将df分开,循环处理
                # 根据'id'列分组
                grouped = df.groupby('id')
                # 循环处理每个分组
                for group_name, group_df in grouped:
                    # 在此处添加你想要执行的处理逻辑，例如：
                    print(group_df)
                    # 将image_numpy转为tif
                    # 除太阳辐射变量 数据类型为unit8,其余变量为float16
                    # 太阳辐射变量
                    image_numpy = utility.get_image_fast(group_df, fields=band, dtype='float')
                    # # 空气温度 土壤水含量
                    # image_numpy = utility.get_image_fast(group_df, fields=band, dtype='float16')
                    print(group_df.columns.values)
                    num_bands = image_numpy.shape[3]
                    # 创建文件驱动
                    driver = gdal.GetDriverByName('GTiff')
                    # # 创建文件驱动
                    for j in range(num_bands):
                        band_data = image_numpy[:, :, :, j]
                        band_data[:,:,0]
                        # out_data = band_data.astype(np.float32)
                        out_data = band_data
                        # 仅针对气象数据 因为气象数据分辨率过高，获取到的数值一个经纬度只有一个值，因而需要重采样，将其转换为500米分辨率，3*3窗口大小的格式，以
                        # 和modis进行匹配
                        # expanded_data = np.repeat(np.repeat(out_data, 3, axis=0), 3, axis=1)
                        # ref PRODUCT : MODIS/061/MCD43A4
                        # ref lai : MODIS/061/MCD15A3H
                        image_name = group_name + '_' + product.replace("/", "_")[10:]
                        image_path = os.path.join(site_image_year_path, image_name)
                        # 检查文件夹是否存在
                        if not os.path.isdir(image_path):
                            # 创建文件夹
                            os.makedirs(image_path)
                            print(f"已创建文件夹：{image_path}")
                        else:
                            print(f"文件夹已存在：{image_path}")

                        # 将影像数据输出为tif
                        file_name = image_name + '_' + band[j] + '.tif'
                        file_path = os.path.join(image_path, file_name)

                        # 将image_numpy的数据写入新文件
                        # dataset = driver.Create(file_path, image_numpy.shape[1], image_numpy.shape[0], 1,gdal.GDT_Float32)
                        # 将image_numpy的数据写入新文件
                        dataset = driver.Create(file_path, image_numpy.shape[1], image_numpy.shape[0], 1,
                                                gdal.GDT_Float32)
                        # 写入仿射变换参数
                        geotransform = get_geotransform(group_df)
                        dataset.SetGeoTransform(geotransform)
                        ## 写入投影
                        # 添加地理转换信息
                        srs = osr.SpatialReference()
                        srs.ImportFromEPSG(3857)  # 设置投影坐标系（EPSG:3857）
                        dataset.SetProjection(srs.ExportToWkt())
                        # 写入数据
                        dataset.GetRasterBand(1).WriteArray(out_data[:, :, 0].copy())
                        dataset.FlushCache()
                        dataset = None
                        # # 针对气象数据
                        # dataset = driver.Create(file_path, 3, 3, 1,
                        #                         gdal.GDT_Float32)
                        # # 写入仿射变换参数
                        # geotransform = get_geotransform(group_df)
                        # dataset.SetGeoTransform(geotransform)
                        # ## 写入投影
                        # # 添加地理转换信息
                        # srs = osr.SpatialReference()
                        # srs.ImportFromEPSG(3857)  # 设置投影坐标系（EPSG:3857）
                        # dataset.SetProjection(srs.ExportToWkt())
                        # # 写入数据
                        # dataset.GetRasterBand(1).WriteArray(expanded_data[:, :, 0].copy())
                        # dataset.FlushCache()
                        # dataset = None
    nodata_df = pd.DataFrame(data=nodata_list)
    nodata_df.to_csv('E:/Carbon_flux/data/nodata/nodata_df_0_10.csv')



