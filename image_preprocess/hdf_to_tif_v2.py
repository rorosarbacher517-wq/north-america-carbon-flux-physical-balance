
import os
# Import packages
import os
import warnings

import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping, box
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep


from osgeo import gdal
import numpy as np
output_daily_single_dir = 'E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif/'
input_files = "E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif/band1.tif"

dataset = gdal.Open("E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif/band1.tif")
print(dataset)

# 构造VRT文件
vrt_options = gdal.BuildVRTOptions(srcNodata=255, separate=True)  # 设置VRT创建选项，包括设置NoData值和将各个输入文件分开

vrt_output_ds = gdal.BuildVRT(output_daily_single_dir, input_files, options=vrt_options)  # 创建VRT数据集

# 将VRT文件导出为一个合并的.tif文件
gdal.Translate("output.tif", vrt_output_ds)

print("VRT文件已生成")
#
# def hdf_subdataset_extraction(hdf_file, dst_dir, subdataset):
#     """unpack a single subdataset from a HDF5 container and write to GeoTiff"""
#     # open the dataset
#     hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
#     band_ds = gdal.Open(hdf_ds.GetSubDatasets()[subdataset][0], gdal.GA_ReadOnly)
#
#     # read into numpy array
#     band_array = band_ds.ReadAsArray().astype(np.int16)
#
#     # convert no_data values
#     # MCD43A4 前7个波段QA波段的No-datahdf文件显示的是 用255代替；后7个波段反射率波段No data 设置为32767 这两个都需要替换为-9999
#     band_array[band_array == 255] = -9999
#
#     # build output path
#     band_path = os.path.join(dst_dir, "band" + str(subdataset+1) + ".tif")
#
#     # write raster
#     out_ds = gdal.GetDriverByName('GTiff').Create(band_path,
#                                                   band_ds.RasterXSize,
#                                                   band_ds.RasterYSize,
#                                                   1,  #Number of bands
#                                                   gdal.GDT_Int16,
#                                                   ['COMPRESS=LZW', 'TILED=YES'])
#     out_ds.SetGeoTransform(band_ds.GetGeoTransform())
#     out_ds.SetProjection(band_ds.GetProjection())
#     out_ds.GetRasterBand(1).WriteArray(band_array)
#     out_ds.GetRasterBand(1).SetNoDataValue(-9999)
#
#     out_ds = None  #close dataset to write to disc
#
# hdf_file  = "E:/Carbon_flux/data/regoin_image/mcd43a4_test/2017-01-10/MCD43A4.A2017010.h02v06.061.2021257230101.hdf"
# dst_dir = 'E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif'
# subdataset = 0
#
# hdf_subdataset_extraction(hdf_file, dst_dir, subdataset)