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
from rasterio.errors import RasterioIOError





if __name__ == '__main__':
    # 下面的功能主要是对fluxnet变量进行聚合，从半小时聚合到天和月，然后是提取一定窗口大小的影像50*50
    # 最终输出要输入模型的x_image_array和y_flux_array
    shp_path = "E:/Carbon_flux/data/regoin_image/AMC_2017_LC_Type1_MCD12Q1_MASK_Points/AMC_2017_LC_Type1_MCD12Q1_MASK_Points.shp"
    # 读取shp文件
    points = gpd.read_file(shp_path)
    # 总的data array，后面每添加一个x_array就添加一个time_array和site_array
    x_list = []
    y_list = []
    # site_image存储根路径
    site_image_path = 'E:/Carbon_flux/data/regoin_image/mcd43a4_test_multiband_tif/'
    # 进入site目录
    sites_list = os.listdir(site_image_path)
    # 进入
    for i in range(0, len(sites_list)):
        print('hhh')
        site = sites_list[i]
        point = points.geometry[i]
        lon = point.x
        lat = point.y
        site_year_path = os.path.join(site_image_path, sites_list[i])
        os.chdir(site_year_path)