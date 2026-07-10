# coding=utf-8
# 利用arcpy进行1.hdf转tif
# 编译器需要切换到D:\arcgis 10.8\python2.7

import os
import arcpy
from arcpy import env


# 设置主文件夹和输出文件夹路径
hdfFolder = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test' # 存储所有MODIS数据的主文件
output_dir = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test_singleband_tif'

# 含有多个波段的hdf转成单波段tif
# 遍历每天的 HDF 文件
hdf_daily_files = os.listdir(hdfFolder)
for i in range(len(hdf_daily_files)):
    # 创建每天的输出文件夹
    output_daily_dir = os.path.join(output_dir, hdf_daily_files[i]).replace('\\', '/')
    if not os.path.exists(output_daily_dir):
        os.makedirs(output_daily_dir)
    # 遍历每天的子文件夹
    input_folder = os.path.join(hdfFolder, hdf_daily_files[i])

    arcpy.env.workspace = input_folder
    hdfList = arcpy.ListRasters('*', 'hdf')

    # band_name = arcpy.Describe(hdfList[0]).children[j].name
    desc = arcpy.Describe(hdfList[0])
    subdatasets = desc.children
    # 逐波段进行处理
    for j in range(0,14):
        if j < 7:
            band_name = 'BRDF_Albedo_Band_Mandatory_Quality_Band'+str(j + 1)  # 确保波段索引从1开始
        else:
            band_name = 'Nadir_Reflectance_Band'+str(j - 6)  # 确保波段索引从1开始
        for hdf in hdfList:
            # print(hdf)
            # 新建波段名文件夹
            band_output_dir = os.path.join(output_daily_dir, band_name).replace('\\', '/')
            # 如果文件夹不存在，则创建一个新的
            if not os.path.exists(band_output_dir):
                os.makedirs(band_output_dir)

            # tif_name = os.path.basename(hdf).replace('hdf', 'tif')
            parts = os.path.basename(hdf).split('.')  # 使用.分割字符串
            tif_name = parts[0] + '_' + parts[2]+'.tif'  # 连接选取的部分并用_进行连接

            output_name = os.path.join(band_output_dir, tif_name) # 单波段格网的名字不能超过13个字符，所以截取其中的几个字符串命名

            # 输出 hdf 同样波段的数据到文件夹下
            data1 = arcpy.ExtractSubDataset_management(hdf, output_name, j)
        print ('第'+str(j)+'个波段处理完成')
    print(str(hdf_daily_files[i])+'天的数据处理完成')

#**********************************************************************************
# 同一个波段的多个hdf转成的tif文件进行镶嵌，合成一个整的tif 合成之后把单个波段的tif文件删除 减少内存
# outpath_multi_tif = r'E:/Carbon_flux/data/regoin_image/mcd43a4_test_multiband_tif'
# inputpath_single_tif = r'E:/Carbon_flux/data/regoin_image/mcd43a4_tif'
#
# singletif_daily_files = os.listdir(inputpath_single_tif)
# for i in range(len(singletif_daily_files)):
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
#         arcpy.MosaicToNewRaster_management(';'.join(tifList), output_daily_single_dir, output_name, pixel_type="32_BIT_FLOAT",
#                                            number_of_bands="1")
#
#         print('Mosaic to new raster complete')






# # hdf转tif
# # 这个产品这一年每一天的文件夹
# hdf_daily_files = os.listdir(hdfFolder)
# multibands = []
# for i in range(len(hdf_daily_files)):
#     # 这个产品某一天的数据存储的路径
#     input_folder = os.path.join(hdfFolder,hdf_daily_files[i])
#     # 逐波段进行处理
#     for j in range(13):
#         # 遍历每天的子文件夹
#         arcpy.env.workspace = input_folder
#         hdfList = arcpy.ListRasters('*', 'hdf')
#         for hdf in hdfList:
#             print(hdf)
#             eviName = os.path.basename(hdf).replace('hdf', 'tif')
#             outname = output_dir + '\\' + hdf_daily_files[i]+eviName
#             print(outname)
#             data1 = arcpy.ExtractSubDataset_management(hdf, outname, j)
#         print('all done' )

# for i in range(13):
#     print i
#
#     sourceDir = u'E:/Carbon_flux/data/regoin_image/mcd43a4_test/2017-01-10/'  # 中文路径 输入
#     targetDir = u'E:/Carbon_flux/data/regoin_image/mcd43a4_tif/'+str(i)+'/'  # 英文路径 输出
#     os.makedirs(targetDir)
#
#     arcpy.CheckOutExtension("Spatial")
#     env.workspace = sourceDir
#     arcpy.env.scratchWorkspace = sourceDir
#     hdfList = arcpy.ListRasters('*', 'hdf')
#     for hdf in hdfList:
#         print hdf
#         eviName = os.path.basename(hdf).replace('hdf', 'tif')
#         outname = targetDir + '\\' + eviName
#         print outname
#         data1 = arcpy.ExtractSubDataset_management(hdf, outname, i)
#     print 'all done'
#     print '------------------------------------------'

# path_list=os.listdir('E:/data/pv_forest_2/land_use/modis_tif/')
# name_list=['LC_Type1','LC_Type2','LC_Type3','LC_Type4','LC_Type5','LC_Prop1_Assessment',
#            'LC_Prop2_Assessment','LC_Prop3_Assessment','LC_Prop1','LC_Prop2','LC_Prop3',
#            'QC','LW']
# for path in path_list:
#     print path
#     i = int(path)
#     arcpy.env.workspace = 'E:/data/pv_forest_2/land_use/modis_tif/'+path
#     tif_list = arcpy.ListRasters("*", "tif")
#     arcpy.MosaicToNewRaster_management(tif_list, 'E:/data/pv_forest_2/land_use/modis_tif_pinjie/', name_list[i]+'_2020.tif',
#                                            pixel_type="16_BIT_SIGNED", number_of_bands=1)



# arcpy.env.workspace = 'E:/data/pv_forest_2/land_use/Esri10/Esri10_2020/'
# tif_list = arcpy.ListRasters("*", "tif")
# arcpy.MosaicToNewRaster_management(tif_list,  'E:/data/pv_forest_2/land_use/Esri10/', 'Esri10_2020.tif',
#                                            pixel_type="16_BIT_SIGNED", number_of_bands=1)

# arcpy.env.workspace = 'E:/data/pv_forest_2/land_use/GLC_FCS2020/GLC_FCS/'
# tif_list = arcpy.ListRasters("*", "tif")
# arcpy.MosaicToNewRaster_management(tif_list,  'E:/data/pv_forest_2/land_use/GLC_FCS2020/', 'GLC_FCS_2020.tif',
#                                            pixel_type="16_BIT_SIGNED", number_of_bands=1)
