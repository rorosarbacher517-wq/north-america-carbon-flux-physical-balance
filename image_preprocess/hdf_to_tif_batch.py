# gdal 实现单个hdf单波段转tif


# import arcpy
# from arcpy import env
# 输入数据文件夹

# 输出数据文件夹

# 设置主文件夹和输出文件夹路径
import os
from osgeo import gdal


def extract_subdatasets(input_hdf, output_tif, band_index):
    # Open the HDF file using GDAL
    hdf_dataset = gdal.Open(input_hdf, gdal.GA_ReadOnly)

    # Get the subdataset based on the band index
    subdataset = hdf_dataset.GetSubDatasets()[band_index]

    # Open the subdataset
    subdataset_dataset = gdal.Open(subdataset[0], gdal.GA_ReadOnly)

    # Read the subdataset as an array
    subdataset_array = subdataset_dataset.ReadAsArray()

    # Get metadata of the subdataset for georeferencing and projection
    geotransform = subdataset_dataset.GetGeoTransform()
    projection = subdataset_dataset.GetProjection()

    # Create the output TIF file and write the array data into it
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_tif, subdataset_dataset.RasterXSize, subdataset_dataset.RasterYSize, 1,
                            gdal.GDT_Float32)
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(subdataset_array)
    outdata.FlushCache()
    outdata = None  # Close the output file


# 设置主文件夹和输出文件夹路径
hdfFolder = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test'  # 存储所有MODIS数据的主文件夹
output_dir = r'E:/Carbon_flux/data/regoin_image/mcd43a4_tif'

# 遍历每天的 HDF 文件
hdf_daily_files = os.listdir(hdfFolder)
for hdf_daily_file in hdf_daily_files:
    # 创建每天的输出文件夹
    output_daily_dir = os.path.join(output_dir, hdf_daily_file).replace('\\', '/')
    if not os.path.exists(output_daily_dir):
        os.makedirs(output_daily_dir)

    # 遍历每天的子文件夹
    input_folder = os.path.join(hdfFolder, hdf_daily_file)
    hdf_files = [file for file in os.listdir(input_folder) if file.endswith('.hdf')]

    for hdf_file in hdf_files:
        hdf_path = os.path.join(input_folder, hdf_file)
        hdf_dataset = gdal.Open(hdf_path, gdal.GA_ReadOnly)

        for band_index in range(14):
            if band_index < 7:
                band_name = 'BRDF_Albedo_Band_Mandatory_Quality_Band' + str(band_index + 1)
            else:
                band_name = 'Nadir_Reflectance_Band' + str(band_index - 6)

            output_band_dir = os.path.join(output_daily_dir, band_name).replace('\\', '/')
            if not os.path.exists(output_band_dir):
                os.makedirs(output_band_dir)

            parts = os.path.basename(hdf_file).split('.')  # 使用.分割字符串
            tif_name = parts[0] + '_' + parts[2] + '.tif'  # 连接选取的部分并用_进行连接

            # output_tif_name = f"{band_name}_{hdf_daily_file}.tif"
            output_tif_path = os.path.join(output_band_dir, tif_name)

            extract_subdatasets(hdf_path, output_tif_path, band_index)

            print(f"Band {band_index} extraction and conversion to TIF completed")

print("All HDF files processed.")
# # 打开一个hdf看看 里面的数据情况
# # 假设input_files中包含了HDF文件的路径列表
# hdf_file_path = input_files[21]  # 假设我们选择第一个HDF文件进行查看
#
# # 打开HDF文件
# hdf_dataset = gdal.Open(hdf_file_path)
#
# # 获取HDF文件中的子数据集数量
# subdataset_count = hdf_dataset.RasterCount
#
# # 获取HDF文件中的子数据集名称
# subdataset_names = [hdf_dataset.GetRasterBand(i + 1).GetDescription() for i in range(subdataset_count)]
#
# print("HDF文件中的子数据集名称：", subdataset_names)
#
# # 选择一个子数据集进行读取
# chosen_subdataset = subdataset_names[0]  # 假设选择第一个子数据集进行读取
# chosen_band = hdf_dataset.GetRasterBand(1)
#
# # 读取子数据集的数据
# data = chosen_band.ReadAsArray()
#
# print("数据示例：", data)

# # 这个产品这一年每一天的文件夹
# hdf_daily_files = os.listdir(hdfFolder)
# for i in range(len(hdf_daily_files)):
#     # # 逐波段进行处理,一共有14个波段
#     # for j in range(13):
#     # 这个产品某一天的数据存储的路径
#     input_folder = os.path.join(hdfFolder,hdf_daily_files[i])
#     multibands = [] # 将每个行列号的数据都存储到这个list 最后转为一个
#     # 遍历每天的子文件夹下的所有行列号的hdf
#     for date_folder in os.listdir(input_folder):
#         date_path = os.path.join(input_folder, date_folder)
#         if os.path.isdir(date_path):
#             output_vrt = os.path.join(output_dir, date_folder + '_merged.vrt')  # 设置输出合并后的VRT文件路径
#             input_files = [os.path.join(date_path, f) for f in os.listdir(date_path) if
#                            f.endswith('.hdf')]  # 获取输入HDF文件的路径列表
#             if len(input_files) > 0:
#                 # 将所有HDF文件合并为单个VRT文件
#                 gdal.BuildVRT(output_vrt, input_files, separate=True)
