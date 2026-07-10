# Initialize earth engine
import ee
service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'E:/Paper_code/biomass-estimates-d58a1f0e77e5.json')
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


# # read the site.csv
# site_df = pd.read_csv("E:/2_biomass/data/sites_reduced_HH_GE_share.csv")
#

import pandas as pd
from osgeo import gdal, osr
import os
import datetime

output_file = 'output.tif'
# 根据经纬度坐标系建立仿射变换矩阵
def get_geotransform(df,driver):
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
    num_bands = 1

    # 创建数据源（Dataset）
    dataset = driver.Create(output_file, width, height, num_bands, gdal.GDT_Float32)

    # 添加地理投影信息
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)  # 设置投影坐标系（EPSG:3857）
    dataset.SetProjection(srs.ExportToWkt())

    # 添加地理变换信息
    xmin, ymin, xmax, ymax = df['longitude_prj'].min(), df['latitude_prj'].min(), df['longitude_prj'].max(), df['latitude_prj'].max()
    pixel_width = (xmax - xmin) / (width - 1)
    pixel_height = (ymax - ymin) / (height - 1)
    geotransform = (xmin, pixel_width, 0, ymax, 0, -pixel_height)
    return geotransform


def save_tif(df,image_numpy,site_image_year_path):
    # 将image_numpy转为tif
    num_bands = image_numpy.shape[3]
    num_dates = image_numpy.shape[2]
    # 创建文件驱动
    driver = gdal.GetDriverByName('GTiff')
    for i in range(num_dates):
        for j in range(num_bands):
            band_data = image_numpy[:, :, i, j]
            band_data = np.expand_dims(band_data, axis=2)
            out_data = band_data.astype(np.float32)
            # 文件名为df['id'] + product（将/转为下划线_）
            image_name = product.replace("/", "_")
            image_path = os.path.join(site_image_year_path, image_name)
            # 检查文件夹是否存在
            if not os.path.isdir(image_path):
                # 创建文件夹
                os.makedirs(image_path)
                print(f"已创建文件夹：{image_path}")
            else:
                print(f"文件夹已存在：{image_path}")
            # 将影像数据输出为tif
            file_name = image_name[8:] + '_' + str(date_list[i]) + '_' + band[j] + '.tif'
            file_path = os.path.join(image_path, file_name)
            dataset = driver.Create(file_path, image_numpy.shape[1], image_numpy.shape[0], 1, gdal.GDT_Float32)
            # 写入仿射变换参数
            geotransform = get_geotransform(df, driver)
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
        return dataset

def get_product_band(year):
    # year = str(year)
    # 将起止日期str转为日期类型
    start_date = str(year) + '-12-01'
    # start_date = datetime.date(*map(int, start_date.split('-')))
    end_date = str(year) + '-13-01'
    # end_date = datetime.date(*map(int, end_date.split('-')))

    # 根据year判断产品类型和band
    if year > 1982 and year < 1994:
        product = [['LANDSAT/LT04/C02/T1_L2'],['LANDSAT/LT04/C02/T2_L2'],['LANDSAT/LT05/C02/T1_L2'],['LANDSAT/LT05/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA']]

    elif year >= 1994 and year < 1999:
        product = [['LANDSAT/LT05/C02/T1_L2'],['LANDSAT/LT05/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA']]
    elif year >= 1999 and year < 2012:
        product = [['LANDSAT/LT05/C02/T1_L2'],['LANDSAT/LT05/C02/T2_L2'],['LANDSAT/LE07/C02/T1_L2'],['LANDSAT/LE07/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA']
                     ]
    elif year == 2012:
        product = [['LANDSAT/LE07/C02/T1_L2'],['LANDSAT/LE07/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA']]
    elif year >= 2013 and year < 2021:
        product = [['LANDSAT/LE07/C02/T1_L2'],['LANDSAT/LE07/C02/T2_L2'],['LANDSAT/LC08/C02/T1_L2'],['LANDSAT/LC08/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL','ST_QA']]
    elif year >= 2021 and year < 2023:
        product = [['LANDSAT/LE07/C02/T1_L2'],['LANDSAT/LE07/C02/T2_L2'],['LANDSAT/LC08/C02/T1_L2'],['LANDSAT/LC08/C02/T2_L2'],['LANDSAT/LC09/C02/T1_L2'],['LANDSAT/LC09/C02/T2_L2']]
        band_list = [['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6', 'QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL','ST_QA'],
                     ['SR_B1','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL','ST_QA']]
    # print(df.head())

    # 根据产品类型选择band
    # band = []
    # for item in product:
    #     if item[0][11] == "4" or item[0][11] == "5" or item[0][11] == "7":
    #         # band.append(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])
    #         band.append(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7','ST_B6','QA_PIXEL'])
    #     elif item[0][11] == "8" or item[0][11] == "9":
    #         # band.append(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])
    #         band.append(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL'])
    return start_date,end_date,product,band_list

def meters_to_lat_lon_displacement(m, origin_lat):
    """
    Rough calculation from meters to lat,lon to calculate buffer sizes
    from https://gis.stackexchange.com/a/2964/96775
    """
    lat = m/111111
    lon = m/(111111 * cos(radians(origin_lat)))
    return lat, lon

def gee_subset_4(product=None,
                   bands=None,
                   start_date=None,
                   end_date=None,
                   latitude=None,
                   longitude=None,
                   scale=None,
                   pad=0,
                   image=None,df = None):
    if pad > 0 :
        pad_lat, pad_lon = meters_to_lat_lon_displacement(pad*1000, origin_lat=latitude)
    # setup the geometry, based upon point locations as specified
    # in the locations file or provided by a latitude or longitude
    # on the command line / when grow is provided pad the location
    # so it becomes a rectangle (report all values raw in a tidy
    # matrix)
    if pad:
        geometry = ee.Geometry.Rectangle([longitude - pad_lon, latitude - pad_lat,
                                          longitude + pad_lon, latitude + pad_lat])
    else:
        geometry = ee.Geometry.Point([longitude, latitude])
    # geometry = ee.Geometry.Point([longitude, latitude])
    col = ee.ImageCollection(product). \
        select(tuple(bands)). \
        filterDate(start_date, end_date)
    try:
        region = col.getRegion(geometry, int(scale)).getInfo()
    except:
        print("Failed to get region, waiting 5 seconds and trying again...")
        t.sleep(5)  # 暂停3秒钟，然后重试
        try:
            region = ee.ImageCollection(ee.Image(product)).getRegion(geometry, int(scale)).getInfo()
        except:
            print("Failed to get region a second time.")
            region = None  # 若要继续下一步操作，region应该设为None或其他默认值
    print('hhhh')
    # stuff the values in a dataframe for convenience
    if region == None:
        return None
    else:
        df = pd.DataFrame.from_records(region[1:len(region)])
        if df.empty:
            return df
        else:
            df.columns = region[0]
            # divide the time field by 1000 as in milliseconds
            # while datetime takes seconds to convert unix time
            # to dates
            df.time = df.time / 1000
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'time': 'date'}, inplace=True)
            df.sort_values(by='date')

            # add the product name and latitude, longitude as a column
            # just to make sense of the returned data after the fact
            df['product'] = pd.Series(product, index=df.index)
    print(df.head())

    # return output
    return df

if __name__ == '__main__':

    sites_df = pd.read_csv("E:/2_biomass/data/Missing_site_data/Missing_data_records.csv")
    # df = pd.read_csv('C:/Users/wuth/Desktop/image_df.csv')
    # print(df.head(5))

    ## 创建影像存储的根目录
    sites_image_root_path = 'E:/2_biomass/data/Missing_site_data/Missing_image'
    nodata_list = []
    nodata_df = pd.DataFrame()
    for m in range(0,len(sites_df)):
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

            # 根据年份判断产品类型和band
            start_date, end_date, product_list, band_list = get_product_band(year)
            print(start_date, end_date, product_list, band_list)

            # 依次遍历产品类型和band 下载影像数据
            for p, b in zip(product_list, band_list):
                product = p[0]
                band = b
                print(product,band)
                #注意区别 4,5,7,8,9的
                if product[11] == '4':
                    df = gee_subset_4(product=product,
                                            bands=band,
                                            start_date= start_date,
                                            end_date= end_date,
                                            latitude= sites_df.loc[m, 'Latitude'],
                                            longitude= sites_df.loc[m, 'Longitude'],
                                            scale=30,
                                           pad = 1.5)
                else:
                    df = gee_subset.gee_subset(product=product,
                                                bands=band,
                                                start_date= start_date,
                                                end_date= end_date,
                                                latitude= sites_df.loc[m, 'Latitude'],
                                                longitude= sites_df.loc[m, 'Longitude'],
                                                scale=30,
                                               pad=1.5)
                    # df = pd.read_csv('C:/Users/wuth/Desktop/image_df.csv')
                    # 在这里应该对df进行排序，确保后面日期和产品数值相对应
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
                        # 将image_numpy转为tif
                        image_numpy = utility.get_image_fast(df, fields=band, dtype=np.uint16)
                        # save_tif(df,image_numpy,site_image_year_path)
                        # 第0维：纬度；第1维：经度；第2维：date；第3维：要下载的波段
                        num_bands = image_numpy.shape[3]
                        num_dates = image_numpy.shape[2]
                        # 创建文件驱动
                        driver = gdal.GetDriverByName('GTiff')
                        for i in range(num_dates):
                            for j in range(num_bands):
                                band_data = image_numpy[:, :, i, j]
                                band_data = np.expand_dims(band_data, axis=2)
                                out_data = band_data.astype(np.float32)
                                # 文件名为df['id'] + product（将/转为下划线_）
                                # image_name = df['id'] + product.replace("/", "_")
                                image_name = id_list[i] + '_' + product.replace("/", "_")[13:]
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
                                dataset = driver.Create(file_path, image_numpy.shape[1], image_numpy.shape[0], 1,
                                                        gdal.GDT_Float32)
                                # 写入仿射变换参数
                                geotransform = get_geotransform(df, driver)
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
    nodata_df = pd.DataFrame(data=nodata_list)
    nodata_df.to_csv('./data/nodata_record/nodata_df_0_50_10.csv')



