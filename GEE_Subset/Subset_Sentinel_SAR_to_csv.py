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
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import Point

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
    # 下载Sentinel-1 SAR数据
    product = ['COPERNICUS/S1_GRD']
    band_list = ['VV','VH','HH','HV']
    return start_date,end_date,product,band_list


def meters_to_lat_lon_displacement(m, origin_lat):
    """
    Rough calculation from meters to lat,lon to calculate buffer sizes
    from https://gis.stackexchange.com/a/2964/96775
    """
    lat = m/111111
    lon = m/(111111 * cos(radians(origin_lat)))
    return lat, lon

def extract_subarray(data):
    # Get the shape of the input array
    height, width, _ = data.shape
    # Calculate the center pixel coordinates
    center_row = height // 2
    center_col = width // 2
    # Calculate the boundaries for the subarray
    start_row = max(center_row - 4, 0)
    end_row = min(center_row + 5, height)
    start_col = max(center_col - 4, 0)
    end_col = min(center_col + 5, width)
    # Extract the subarray
    subarray = data[start_row:end_row, start_col:end_col]
    return subarray


if __name__ == '__main__':
    sites_df = pd.read_csv("E:/Sentinel SAR/2023_30m_cdls_11SKA_samples_locs.csv")
    # df = pd.read_csv('C:/Users/wuth/Desktop/image_df.csv')
    # df = pd.read_csv("E:/Carbon_flux/data/input_data/MODIS_Sites_test.csv")
    # print(df.head(5))
    ## 创建影像存储的根目录
    sites_image_root_path = 'E:/Sentinel SAR/SKA_samples_locs_images/'
    # sites_image_root_path = 'E:\Carbon_flux\data\modis\Sites_image_ERA5_LAND_DAILY_AGGR'
    nodata_list = []
    nodata_df = pd.DataFrame()
    # for m in range(158,len(sites_df)):
    # 打印开始时间
    start_time = datetime.datetime.now()
    print("开始时间:", start_time)
    for m in range(1,2):
        sites_image_path_name = str(sites_df['id'][m])
        sites_image_path = os.path.join(sites_image_root_path, sites_image_path_name)
        # 检查文件夹是否存在
        if not os.path.isdir(sites_image_path):
            # 创建文件夹
            os.makedirs(sites_image_path)
        # year_list = []
        for year in range((sites_df['StartYear'][m]), (sites_df['EndYear'][m] + 1)):
            site_image_year_name = str(year)
            site_image_year_path = os.path.join(sites_image_path, site_image_year_name)
            # 检查文件夹是否存在
            if not os.path.isdir(site_image_year_path):
                # 创建文件夹
                os.makedirs(site_image_year_path)
            start_date, end_date, product_list, band_list = get_product_band(year)
            orbit_list = ['ASCENDING','DESCENDING']
            product = product_list[0]
            for orbit in orbit_list:
                for band in band_list:
                    df = gee_subset.gee_subset(product=product,
                                               bands=[band],
                                               instrument='IW',
                                               orbit = orbit,
                                               start_date=start_date,
                                               end_date=end_date,
                                               latitude = sites_df.loc[m, 'Latitude'],
                                               longitude = sites_df.loc[m, 'Longitude'],
                                               scale=10,
                                               pad=0.05)
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
                            # 将image_numpy转为tif
                            # image_numpy = utility.get_image_fast(group_df, fields=[band], dtype='float')
                            image_numpy, lon_result, lat_result = utility.get_image_fast(group_df, fields=[band], dtype='float')
                            lat_result[:,:,0,0]
                            # image_numpy = utility.get_image_fast(group_df, fields=band, dtype='float16')
                            # print(group_df.columns.values)
                            # 获取所需要的43*43大小的影像块
                            band_data = extract_subarray(image_numpy[:,:,:,0])
                            band_lon = extract_subarray(lon_result[:, :, :, 0])
                            band_lat = extract_subarray(lat_result[:, :, :, 0])
                            # 打印仿射变换参数
                            # print(transform)
                            image_name = group_name
                            image_path = os.path.join(site_image_year_path, image_name)
                            # 检查文件夹是否存在
                            if not os.path.isdir(image_path):
                                # 创建文件夹
                                os.makedirs(image_path)
                                # print(f"已创建文件夹：{image_path}")
                            # 将影像数据输出为tif
                            file_name = image_name + '_' + band + '.csv'
                            file_path = os.path.join(image_path, file_name)

                            band_df = pd.DataFrame(band_data[:,:,0])
                            band_df.to_csv(file_path)
                            # print(f"已保存影像数据至：{file_path}")
    # nodata_df = pd.DataFrame(data=nodata_list)
    # nodata_df.to_csv('E:/Carbon_flux/data/nodata/nodata_df_0_10.csv')
    # 循环结束后的时间
    end_time = datetime.datetime.now()
    print("结束时间:", end_time)
    # 计算时间差
    time_difference = end_time - start_time
    print("经过的时间:", time_difference)



