# coding=utf-8
# 利用arcpy进行1.hdf转tif
# 编译器需要切换到

import os
# import arcpy
from osgeo import gdal
import os


#**********************************************************************************
# 同一个波段的多个hdf转成的tif文件进行镶嵌，合成一个整的tif 合成之后把单个波段的tif文件删除 减少内存
# outpath_multi_tif = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test_multiband_tif'
# inputpath_single_tif = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif'

input_folder = "E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif"  # 替换为你的输入文件夹路径
output_vrt_file = "E:/Carbon_flux/data/regoin_image/mcd43a4_tif_multiband"  # 替换为你的输出VRT文件路径

def batch_raster_mosaic_with_vrt(input_folder, output_vrt_file):
    import glob
    singletif_daily_files = os.listdir(input_folder)
    # # gdal拼接同一个文件夹下的tif
    for i in range(len(singletif_daily_files)): # len(singletif_daily_files)
        # 创建每天的输出文件夹
        output_daily_single_dir = os.path.join(output_vrt_file, singletif_daily_files[i]).replace('\\', '/')
        if not os.path.exists(output_daily_single_dir):
            os.makedirs(output_daily_single_dir)
        # 遍历每天的子文件夹
        daily_input_folder = os.path.join(input_folder, singletif_daily_files[i])
        band_list = os.listdir(daily_input_folder)
        for j in range(len(band_list)):
            band_path = os.path.join(daily_input_folder, band_list[j])
            # Get all the .tif files in the input folder
            input_files = [os.path.join(band_path, file).replace('\\', '/') for file in os.listdir(band_path) if
                           file.endswith('.tif')]

            # Remove empty tif files
            input_files = [file for file in input_files if gdal.Open(file) is not None]

            if len(input_files) == 0:
                print("不存在非空tif文件，拼接失败")
                continue

            vrt_output_ds = gdal.BuildVRT('', input_files)  # 创建VRT数据集，不保存到本地

            # 分块保存
            for block_index in range(4):
                block_output_path = os.path.join(output_daily_single_dir, f"{band_list[j]}_block_{block_index}.tif")
                gdal.Translate(block_output_path, vrt_output_ds,
                               srcWin=[0, block_index * vrt_output_ds.RasterYSize // 4, vrt_output_ds.RasterXSize,
                                       vrt_output_ds.RasterYSize // 4])

            print("VRT文件已生成")

# def batch_raster_mosaic_with_vrt(input_folder, output_vrt_file):
#     import glob
#     singletif_daily_files = os.listdir(input_folder)
#     # # gdal拼接同一个文件夹下的tif
#     for i in range(len(singletif_daily_files)): # len(singletif_daily_files)
#         # 创建每天的输出文件夹
#         output_daily_single_dir = os.path.join(output_vrt_file, singletif_daily_files[i]).replace('\\', '/')
#         if not os.path.exists(output_daily_single_dir):
#             os.makedirs(output_daily_single_dir)
#         # 遍历每天的子文件夹
#         daily_input_folder = os.path.join(input_folder, singletif_daily_files[i])
#         band_list = os.listdir(daily_input_folder)
#         for j in range(len(band_list)):
#             band_path = os.path.join(daily_input_folder, band_list[j])
#             # Get all the .tif files in the input folder
#             input_files = [os.path.join(band_path, file).replace('\\','/') for file in os.listdir(band_path) if file.endswith('.tif')]
#
#             # Remove empty tif files
#             input_files = [file for file in input_files if gdal.Open(file) is not None]
#
#             if len(input_files) == 0:
#                 print("不存在非空tif文件，拼接失败")
#                 continue
#
#             vrt_output_ds = gdal.BuildVRT(output_daily_single_dir, input_files)  # 创建VRT数据集
#
#             # 将VRT文件导出为一个合并的.tif文件
#             output_path = os.path.join(output_daily_single_dir,band_list[j]) + '.tif'
#             gdal.Translate(output_path, vrt_output_ds)
#
#             print("VRT文件已生成")

# 使用示例
batch_raster_mosaic_with_vrt(input_folder, output_vrt_file)
#
# # **********************************************
# def mosaic_tif_files(input_folder, output_file):
#     singletif_daily_files = os.listdir(input_folder)
#     # # gdal拼接同一个文件夹下的tif
#     for i in range(0, 1):
#         # 创建每天的输出文件夹
#         output_daily_single_dir = os.path.join(input_folder, singletif_daily_files[i]).replace('\\', '/')
#         if not os.path.exists(output_daily_single_dir):
#             os.makedirs(output_daily_single_dir)
#         # 遍历每天的子文件夹
#         daily_input_folder = os.path.join(inputpath_single_tif, singletif_daily_files[i])
#         band_list = os.listdir(daily_input_folder)
#         for j in range(len(band_list)):
#             band_path = os.path.join(daily_input_folder,band_list[j])
#             # 获取输入文件夹中的所有tif文件
#             input_files = [file for file in os.listdir(band_path) if file.endswith('.tif')]
#
#             # 移除空tif文件
#             input_files = [file for file in input_files if gdal.Open(os.path.join(band_path, file)).ReadAsArray() is not None]
#
#             if len(input_files) == 0:
#                 print("不存在非空tif文件，拼接失败")
#                 continue
#
#             # 读取第一个非空tif文件，获取其信息
#             first_tif = gdal.Open(os.path.join(band_path, input_files[0]))
#             driver = gdal.GetDriverByName("GTiff")
#
#             # 创建输出文件
#             output_dataset = driver.Create(output_daily_single_dir, first_tif.RasterXSize, first_tif.RasterYSize,
#                                            len(input_files),
#                                            first_tif.GetRasterBand(1).DataType)
#             # 将各个非空tif文件写入到输出文件中
#             for k in range(len(input_files)):
#                 current_tif = gdal.Open(os.path.join(band_path, input_files[k]))
#                 output_dataset.GetRasterBand(k + 1).WriteArray(current_tif.ReadAsArray())
#                 output_dataset.GetRasterBand(k + 1).SetNoDataValue(current_tif.GetRasterBand(1).GetNoDataValue())
#
#             # 设置地理参考信息
#             output_dataset.SetGeoTransform(first_tif.GetGeoTransform())
#             output_dataset.SetProjection(first_tif.GetProjection())
#             output_dataset.FlushCache()
#
#             # 关闭所有数据集
#             del output_dataset
#
# mosaic_tif_files(inputpath_single_tif, outpath_multi_tif)
# #
# # # gdal拼接同一个文件夹下的tif
# for i in range(0,1):
#     # 创建每天的输出文件夹
#     output_daily_single_dir = os.path.join(outpath_multi_tif, singletif_daily_files[i]).replace('\\', '/')
#     if not os.path.exists(output_daily_single_dir):
#         os.makedirs(output_daily_single_dir)
#     # 遍历每天的子文件夹
#     daily_input_folder = os.path.join(inputpath_single_tif, singletif_daily_files[i])
#     # 进入各个波段
#     band_list = os.listdir(daily_input_folder)
#     for j in range(len(band_list)):
#         # 依次进入各个波段进行镶嵌合成
#         input_folder = os.path.join(daily_input_folder,band_list[j])
#         # 将singletif_path下的所有tif镶嵌成一个 并以input_folder[j]命名 存储到output_daily_single_dir文件夹下
#         # 获取输入文件夹中的所有tif文件
#         input_files = [file for file in os.listdir(input_folder) if file.endswith('.tif')]
#         # 移除空tif文件
#         input_files = [file for file in input_files if gdal.Open(os.path.join(input_folder, file)).RasterXSize > 0]
#
#         if len(input_files) == 0:
#             print("不存在非空tif文件，拼接失败")
#             return
#
#         # 读取第一个tif文件，获取其信息
#         first_tif = gdal.Open(os.path.join(input_folder, input_files[37]))
#         driver = gdal.GetDriverByName("GTiff")
#
#         # 创建输出文件
#         output_dataset = driver.Create(output_daily_single_dir, first_tif.RasterXSize, first_tif.RasterYSize, len(input_files),
#                                        first_tif.GetRasterBand(1).DataType)
#         # 将各个tif文件写入到输出文件中
#         for i in range(len(input_files)):
#             current_tif = gdal.Open(os.path.join(input_folder, input_files[i]))
#             output_dataset.GetRasterBand(i + 1).WriteArray(current_tif.ReadAsArray())
#             output_dataset.GetRasterBand(i + 1).SetNoDataValue(current_tif.GetRasterBand(1).GetNoDataValue())
#
#         # 设置地理参考信息
#         output_dataset.SetGeoTransform(first_tif.GetGeoTransform())
#         output_dataset.SetProjection(first_tif.GetProjection())
#         output_dataset.FlushCache()
#
#         # 关闭所有数据集
#         del output_dataset



# #***********************************
# # arcpy实现
# # len(singletif_daily_files)
# for i in range(0,1):
#     # 创建每天的输出文件夹
#     output_daily_single_dir = os.path.join(outpath_multi_tif, singletif_daily_files[i]).replace('\\', '/')
#     if not os.path.exists(output_daily_single_dir):
#         os.makedirs(output_daily_single_dir)
#     # 遍历每天的子文件夹
#     input_folder = os.path.join(inputpath_single_tif, singletif_daily_files[i])
#     # 进入各个波段
#     band_list = os.listdir(input_folder)
#     for j in range(len(band_list)):
#         # 依次进入各个波段进行镶嵌合成
#         singletif_path = os.path.join(input_folder,band_list[j])
#         # 将singletif_path下的所有tif镶嵌成一个 并以input_folder[j]命名 存储到output_daily_single_dir文件夹下
#
#         # 列出路径下的所有tif文件
#         arcpy.env.workspace = singletif_path
#         tifList = arcpy.ListRasters('*', 'tif')
#         # tifList = arcpy.ListRasters('*', 'tif', singletif_path)
#
#         # 设置输出多波段tif文件的名称
#         output_name = band_list[j] # 请将"path_to_output_multiband_tif"替换为您希望输出的多波段tif文件路径
#
#         # 使用MosaicToNewRaster_management函数进行镶嵌
#         # arcpy.MosaicToNewRaster_management(';'.join(tifList), output_daily_single_dir, output_name, pixel_type="32_BIT_FLOAT",
#         #                                    number_of_bands="1")
#         # 使用MosaicToNewRaster_management进行镶嵌，并设置投影
#         arcpy.MosaicToNewRaster_management(tifList, output_daily_single_dir, output_name,
#                                            pixel_type="32_BIT_FLOAT",
#                                            number_of_bands=1)
#         print('Mosaic to new raster complete')
#
