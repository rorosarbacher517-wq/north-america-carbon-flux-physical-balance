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


# def get_product_band(year):
#     # year = str(year)
#     # 将起止日期str转为日期类型
#     start_date = str(year) + '-01-01'
#     # start_date = datetime.date(*map(int, start_date.split('-')))
#     end_date = str(year+1) + '-01-01'
#
#     # end_date = datetime.date(*map(int, end_date.split('-')))
#     # # 下载影像格式的气象数据 根据year判断产品类型和band
#     # if year > 1999 and year < 2022:
#     #     product = ['ECMWF/ERA5_LAND/DAILY_AGGR']
#     #     band_list = ['surface_solar_radiation_downwards_sum','temperature_2m','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']
#     #     # band_list = ['surface_solar_radiation_downwards_sum']
#     # else:
#     #     print('the data is not available')
#     if year > 1999 and year < 2025:
#         product = ['MODIS/061/MCD43A4']
#         band_list = ['BRDF_Albedo_Band_Mandatory_Quality_Band1','BRDF_Albedo_Band_Mandatory_Quality_Band2','BRDF_Albedo_Band_Mandatory_Quality_Band3',
#                      'BRDF_Albedo_Band_Mandatory_Quality_Band4','BRDF_Albedo_Band_Mandatory_Quality_Band5','BRDF_Albedo_Band_Mandatory_Quality_Band6',
#                      'BRDF_Albedo_Band_Mandatory_Quality_Band7',
#                      'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3',
#                      'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7']
#     else:
#         print('the data is not available')
#     # # 下载LAi数据
#     # if year > 2001 and year < 2023:
#     #     product = ['MODIS/061/MCD15A3H']
#     #     band_list = ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC','FparStdDev','LaiStdDev']
#     # else:
#     #     print('the data is not available')
#     # if year > 1999 and year < 2023:
#     #     product = ['MODIS/061/MOD15A2H']
#     #     band_list = ['Fpar_500m','Lai_500m','FparLai_QC','FparExtra_QC','FparStdDev_500m','LaiStdDev_500m']
#     # else:
#     #     print('the data is not available')
#     return start_date,end_date,product,band_list

def get_product_band(year):
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"

    product = None
    band_list = None

    if year > 2011 and year < 2025:
        # product = ['MODIS/061/MCD43A4']
        # band_list = [
        #     'BRDF_Albedo_Band_Mandatory_Quality_Band1', 'BRDF_Albedo_Band_Mandatory_Quality_Band2',
        #     'BRDF_Albedo_Band_Mandatory_Quality_Band3', 'BRDF_Albedo_Band_Mandatory_Quality_Band4',
        #     'BRDF_Albedo_Band_Mandatory_Quality_Band5', 'BRDF_Albedo_Band_Mandatory_Quality_Band6',
        #     'BRDF_Albedo_Band_Mandatory_Quality_Band7', 'Nadir_Reflectance_Band1',
        #     'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4',
        #     'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6', 'Nadir_Reflectance_Band7'
        # ]
        # product = ['MODIS/061/MCD15A3H']
        # band_list = ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC', 'FparStdDev', 'LaiStdDev']
        product = ['NASA/VIIRS/002/VNP43IA4']
        band_list = ['Nadir_Reflectance_I1', 'Nadir_Reflectance_I2', 'Nadir_Reflectance_I3', 'BRDF_Albedo_Band_Mandatory_Quality_I1', 'BRDF_Albedo_Band_Mandatory_Quality_I2', 'BRDF_Albedo_Band_Mandatory_Quality_I3']
    else:
        print(f"[跳过] {year}年不在有效范围 (2000–2024)")

    return start_date, end_date, product, band_list

def meters_to_lat_lon_displacement(m, origin_lat):
    """
    Rough calculation from meters to lat,lon to calculate buffer sizes
    from https://gis.stackexchange.com/a/2964/96775
    """
    lat = m/111111
    lon = m/(111111 * cos(radians(origin_lat)))
    return lat, lon


if __name__ == '__main__':
    sites_df = pd.read_csv("E:/HLS carbon flux/data/Sites data/MODIS time series/sites_list_csv/AFO_total.csv")
    # df = pd.read_csv('C:/Users/wuth/Desktop/image_df.csv')
    # df = pd.read_csv("E:/Carbon_flux/data/input_data/MODIS_Sites_test.csv")
    # print(df.head(5))
    ## 创建影像存储的根目录
    sites_image_root_path = 'E:/HLS carbon flux/data/Sites data/MODIS time series/VNP43IA4'
    # sites_image_root_path = 'E:\Carbon_flux\data\modis\Sites_image_ERA5_LAND_DAILY_AGGR'
    nodata_list = []
    nodata_df = pd.DataFrame()
    # for m in range(158,len(sites_df)):
    for m in range(35,70):
        site_id = sites_df['Site_Id'][m]
        site_lat = sites_df['Latitude'][m]
        site_lon = sites_df['Longitude'][m]
        site_path = os.path.join(sites_image_root_path, site_id)

        # 检查文件夹是否存在
        if not os.path.isdir(site_path):
            # 创建文件夹
            os.makedirs(site_path)
            print(f"已创建文件夹：{site_path}")
        else:
            print(f"文件夹已存在：{site_path}")
        # year_list = []
        for year in range((sites_df['StartYear'][m]), (sites_df['EndYear'][m] + 1)):
            year_path = os.path.join(site_path, str(year))
            os.makedirs(year_path, exist_ok=True)
            site_image_year_name = str(year)
            site_image_year_path = os.path.join(site_path, site_image_year_name)
            # 检查文件夹是否存在
            if not os.path.isdir(site_image_year_path):
                # 创建文件夹
                os.makedirs(site_image_year_path)
                print(f"已创建文件夹：{site_image_year_path}")
            else:
                print(f"文件夹已存在：{site_image_year_path}")

            start_date, end_date, product_list, band_list = get_product_band(year)
            # 如果数据不可用，跳过该年
            if product_list is None or band_list is None:
                continue
            # print(start_date, end_date, product_list, band_list)

            # if not product_list:
            #     continue
            product = product_list[0]

            # 获取影像数据
            df = gee_subset.gee_subset(product=product,
                                       bands=band_list,
                                       start_date=start_date,
                                       end_date=end_date,
                                       latitude=sites_df.loc[m, 'Latitude'],
                                       longitude=sites_df.loc[m, 'Longitude'],
                                       scale=500,
                                       pad=1.5)

            if df is None or df.empty:
                nodata_name = f"{site_id}_{year}_{product.replace('/', '_')}"
                nodata_list.append(nodata_name)
                nodata_df.loc[m, 'site'] = site_id
                nodata_df.loc[m, 'nodata'] = nodata_name
                continue

            df.sort_values(by='id', inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df['formatted_date'] = df['date'].dt.strftime('%Y_%m_%d')
            df.fillna(value=-9999, inplace=True)

            grouped = df.groupby('id')

            for group_name, group_df in grouped:
                image_numpy = utility.get_image_fast(group_df, fields=band_list, dtype='float')
                num_bands = image_numpy.shape[3]
                image_name = f"{group_name}_{product.replace('/', '_')[10:]}"
                image_path = os.path.join(year_path, image_name)

                # 如果所有波段文件都存在则跳过
                if os.path.isdir(image_path):
                    all_exist = all(
                        os.path.exists(os.path.join(image_path, f"{image_name}_{band}.tif"))
                        for band in band_list
                    )
                    if all_exist:
                        print(f"[跳过] 所有波段已存在：{image_name}")
                        continue
                else:
                    os.makedirs(image_path, exist_ok=True)

                # 写入每个波段
                for j, band in enumerate(band_list):
                    file_name = f"{image_name}_{band}.tif"
                    file_path = os.path.join(image_path, file_name)

                    if os.path.exists(file_path):
                        print(f"[跳过] 已存在：{file_name}")
                        continue

                    out_data = image_numpy[:, :, :, j]
                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(file_path, out_data.shape[1], out_data.shape[0], 1, gdal.GDT_Float32)

                    geotransform = get_geotransform(group_df)
                    dataset.SetGeoTransform(geotransform)

                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(3857)
                    dataset.SetProjection(srs.ExportToWkt())

                    dataset.GetRasterBand(1).WriteArray(out_data[:, :, 0].copy())
                    dataset.FlushCache()
                    dataset = None

                    print(f"[保存成功] {file_name}")

    # 保存无数据站点信息
    nodata_df_out = pd.DataFrame(data=nodata_list, columns=['nodata_info'])
    nodata_df_out.to_csv('E:/Carbon_flux/data/nodata/nodata_df_1_60.csv', index=False)



