import numpy as np
import os, argparse, re
from math import cos, radians
import pandas as pd
from math import radians, degrees, asin, sin, cos, sqrt

from pyproj import Transformer
import rasterio
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pytz
from pytz import timezone
from datetime import datetime

df = pd.read_csv("E:/df.csv")
# # 将时间字段除以1000转换为秒级别的Unix时间戳
# df['time'] = df['time'] / 1000
#
# 将时间字段除以1000转换为秒级别的Unix时间戳
df['time'] = df['time'] / 1000

# 根据经纬度获取所在时区
# 根据经纬度获取所在时区
def get_timezone(lon, lat):
    # 这里使用一个简单的示例，实际情况可能需要使用更复杂的方法来获取时区
    if lon > 0:
        return timezone('Etc/GMT+' + str(int(lon / 15)))
    else:
        return timezone('Etc/GMT-' + str(int(abs(lon) / 15)))

# 将时间戳转换为所在时区对应的UTC时间并替换原time列
df['time'] = df.apply(lambda row: datetime.utcfromtimestamp(row['time']).replace(tzinfo=pytz.utc).astimezone(get_timezone(row['longitude'], row['latitude'])), axis=1)

# 将转换后的时间赋值给新的'date'列并删除原time列
df['date'] = df['time']
df.drop(columns='time', inplace=True)

# 添加'product'列
# df['product'] = pd.Series(product, index=df.index)
print(df.head())
# # 将儒略日转换为日期
# julian_date = 20160303
# reference_date = datetime(4713, 1, 1)
# target_date = reference_date + timedelta(days=julian_date)
#
# # 打印转换后的日期（本地时间）
# print("Local Time (New York):", target_date)
#
# # 将本地时间转换为UTC时间
# utc_target_date = target_date - timedelta(hours=5)  # 美国纽约所在东部时区为UTC-5
# print("UTC Time:", utc_target_date)
# from datetime import datetime
# import pytz
#
# # 定义纽约时区
# ny_timezone = pytz.timezone('America/New_York')
#
# # 创建本地时间对象
# local_time = datetime(2016, 3, 3)
#
# # 将本地时间转换为纽约时区时间
# local_time_ny = ny_timezone.localize(local_time)
#
# # 将本地时间转换为UTC时间
# utc_time = local_time_ny.astimezone(pytz.utc)
#
# print("Local Time (New York):", local_time_ny)
# print("UTC Time:", utc_time)

# def get_geotransform(df):
#     # 将经纬度转换为3857投影坐标
#     transformer = Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
#     df['longitude_prj'], df['latitude_prj'] = transformer.transform(df['longitude'].values, df['latitude'].values)
#
#     # 计算仿射变换参数
#     xres = (df['longitude_prj'].max() - df['longitude_prj'].min()) / 500
#     yres = (df['latitude_prj'].max() - df['latitude_prj'].min()) / 500
#     xmin = df['longitude_prj'].min() + xres/2 - 250
#     ymax = df['latitude_prj'].max() - yres/2 + 250
#
#     geotransform = (xmin, xres, 0, ymax, 0, -yres)
#     return geotransform
#
# def save_as_geotiff(df, geotransform, output_file):
#     # 创建一个空的数组来存储像素值
#     arr = np.zeros((500, 500), dtype=np.uint16)  # 假设数据是16位整数型
#
#     # 设置地理参考信息
#     kwargs = {
#         'count': 1,
#         'crs': 'EPSG:3857',
#         'width': 500,
#         'height': 500,
#         'transform': geotransform
#     }
#
#     # 将数组保存为geotiff文件
#     with rasterio.open(output_file, 'w', **kwargs) as dst:
#         dst.write(arr, 1)
#
# df = pd.read_csv("E:/Carbon_flux/data/input_data/MODIS_Sites_test_200002.csv")
# # 调用函数并保存为geotiff文件
# geotransform = get_geotransform(df)
# save_as_geotiff(df, geotransform, 'output_image.tif')