import os
import geopandas as gpd
from osgeo import gdal, osr
import numpy as np

# 1. 读取矢量点图层
shapefile_path = "E:/zzj_road/data/point1/test1Pro.shp"
points = gpd.read_file(shapefile_path)

print(f"Total features in the Shapefile: {len(points)}")
print(points)

# 2. 定义影像文件路径的列表
tif_folder = 'E:/zzj_road/data/11/11'
tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')]
print(f"Total TIF files: {len(tif_files)}")

# 3. 定义输出文件夹路径
output_folder = r'E:/zzj_road/data/11/11_crop'
os.makedirs(output_folder, exist_ok=True)

# 4. 遍历每个点，处理对应的影像
for index, row in points.iterrows():
    point = row.geometry
    point_x, point_y = point.x, point.y

    for tif_path in tif_files:
        dataset = gdal.Open(tif_path)
        geo_transform = dataset.GetGeoTransform()

        # 计算影像的边界
        x_min = geo_transform[0]
        x_max = x_min + geo_transform[1] * dataset.RasterXSize
        y_min = geo_transform[3] + geo_transform[5] * dataset.RasterYSize
        y_max = geo_transform[3]

        # 检查点是否在影像范围内
        if x_min < point_x < x_max and y_min < point_y < y_max:
            # 计算像素坐标
            px = int((point_x - geo_transform[0]) / geo_transform[1])
            py = int((point_y - geo_transform[3]) / geo_transform[5])

            # 定义裁剪窗口
            half_window_size = 510
            window_size = 1021
            window_x_min = max(px - half_window_size, 0)
            window_y_min = max(py - half_window_size, 0)
            window_x_max = min(px + half_window_size, dataset.RasterXSize)
            window_y_max = min(py + half_window_size, dataset.RasterYSize)

            # 裁剪影像
            clipped_image = dataset.ReadAsArray(window_x_min, window_y_min, window_x_max - window_x_min, window_y_max - window_y_min)

            # 保存裁剪结果
            output_tif_path = os.path.join(output_folder, f'output_cropped_{index}.tif')
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(output_tif_path, window_size, window_size, dataset.RasterCount, gdal.GDT_Float32)

            # 设置投影和地理变换
            out_geo_transform = (
                geo_transform[0] + (window_x_min * geo_transform[1]),
                geo_transform[1],
                geo_transform[2],
                geo_transform[3] + (window_y_min * geo_transform[5]),
                geo_transform[4],
                geo_transform[5]
            )
            out_dataset.SetGeoTransform(out_geo_transform)

            # 设置投影为 EPSG:4326
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            out_dataset.SetProjection(srs.ExportToWkt())

            for i in range(dataset.RasterCount):
                out_dataset.GetRasterBand(i + 1).WriteArray(clipped_image[i])

            out_dataset.FlushCache()
            out_dataset = None
            break


# import os
# import geopandas as gpd
# import rasterio
# from rasterio.windows import Window
#
# # 1. 读取矢量点图层
# shapefile_path = 'C:/Users/91390/Desktop/point1/test1Pro.shp'
# points = gpd.read_file(shapefile_path)
#
# print(f"Total features in the Shapefile: {len(points)}")
# print(points)
#
# # 2. 定义影像文件路径的列表
# tif_folder = 'C:/Users/91390/Desktop/11'
# tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')]
# print(f"Total TIF files: {len(tif_files)}")
#
# # 3. 定义输出文件夹路径
# output_folder = r'C:/Users/91390/Desktop/11'
# os.makedirs(output_folder, exist_ok=True)
#
# # 4. 遍历每个点，处理对应的影像
# for index, row in points.iterrows():
#     point = row.geometry
#     point_x, point_y = point.x, point.y
#
# for tif_path in tif_files:
#     with rasterio.open(tif_path) as src:
#         if src.bounds.left < point_x < src.bounds.right and src.bounds.bottom < point_y < src.bounds.top:
#             px, py = src.index(point_x, point_y)
#             print(px)
#             print(py)
#
#     # 定义裁剪窗口
#     half_window_size = 510
#     window = Window(px - half_window_size, py - half_window_size, 1021, 1021)
#
#     # 裁剪影像
#     clipped_image = src.read(window=window)
#
#     # 保存裁剪结果
#     output_tif_path = os.path.join(output_folder, f'output_cropped_{index}.tif')
#     profile = src.profile
#     profile.update({
#     'width': 1021,
#     'height': 1021,
#     'transform': rasterio.windows.transform(window, src.transform),
#     'crs': src.crs # 使用影像的 CRS
#     })
#
#     with rasterio.open(output_tif_path, 'w', **profile) as dst:
#         dst.write(clipped_image)
#
#         break