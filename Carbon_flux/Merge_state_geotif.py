import os
from osgeo import gdal
import numpy as np
import os
import rasterio
from rasterio.merge import merge
from scipy.ndimage import generic_filter
from osgeo import gdal
from rasterio.windows import Window

# input_folder = "E:/The North America ecosystem carbon flux/Region_data/block_geotifs_merge_to_state1/"
# output_folder = "E:/The North America ecosystem carbon flux/Region_data/state_geotifs/"
#
# # 获取./block_geotifs_merge_to_state/文件夹下的所有州文件夹
# state_folders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
#
# # 初始化空列表来存储每个州文件夹下的GPP、NEE、RECO文件路径
# gpp_files = []
# nee_files = []
# reco_files = []
#
# # 遍历每个州文件夹
# for state_folder in state_folders:
#     gpp_files.append(os.path.join(state_folder, "GPP_RLM.tif"))
#     nee_files.append(os.path.join(state_folder, "NEE_RLM.tif"))
#     reco_files.append(os.path.join(state_folder, "RECO_RLM.tif"))
#
# # 合并所有GPP_RLM.tif文件
# gdal.Warp(os.path.join(output_folder, "GPP.tif"), gpp_files, format="GTiff")
#
# # 合并所有NEE_RLM.tif文件
# gdal.Warp(os.path.join(output_folder, "NEE.tif"), nee_files, format="GTiff")
#
# # 合并所有RECO_RLM.tif文件
# gdal.Warp(os.path.join(output_folder, "RECO.tif"), reco_files, format="GTiff")
#
#
#
#
region_base_dir = 'E:/The North America ecosystem carbon flux/Region_data/State_mosaci_geotifs/0216/'

# 进入国家级下面的州级目录
nation_list = os.listdir(region_base_dir)
nation_path = os.path.join(region_base_dir, 'State_clip')
out_dir = os.path.join(region_base_dir, 'Daily')

# 进入国家级下面的州级目录
#
output_files = ['NEE_RLM_clip.tif', 'GPP_RLM_clip.tif', 'RECO_RLM_clip.tif']
#  ************************
# 假设 output_files 和 nation_path, out_dir 已经定义
for variable_name in output_files:
    # Read various tif files corresponding to each variable
    variable_files = [file for file in os.listdir(nation_path) if file.endswith(variable_name)]

    # Create a list of opened rasterio file objects
    src_files_to_mosaic = [rasterio.open(os.path.join(nation_path, file)) for file in variable_files]

    # Merge the raster files
    mosaic, out_trans = merge(src_files_to_mosaic)

    # # Multiply the mosaic data by 365
    # mosaic = mosaic * 365

    # Set the path for the merged output file
    output_file = os.path.join(out_dir, variable_name)
    print(f"Merged file saved at {output_file}")

    # Write the mosaic to the output file
    with rasterio.open(output_file, 'w', driver='GTiff', width=mosaic.shape[2], height=mosaic.shape[1], count=1,
                       dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dst:
        dst.write(mosaic)


##  重叠部分取第一张
# region_base_dir = 'E:/The North America ecosystem carbon flux/Region_data/'
#
# # 进入国家级下面的州级目录
# nation_path = os.path.join(region_base_dir, 'State_clip')
# out_dir = os.path.join(region_base_dir, 'State_mosaci_geotifs')
#
# output_files = ['NEE_RLM_clip.tif', 'GPP_RLM_clip.tif', 'RECO_RLM_clip.tif']
#
# for variable_name in output_files:
#     # Read various tif files corresponding to each variable
#     variable_files = [file for file in os.listdir(nation_path) if file.endswith(variable_name)]
#
#     # Create a list of opened rasterio file objects
#     src_files_to_mosaic = [rasterio.open(os.path.join(nation_path, file)) for file in variable_files]
#
#     # Merge the raster files
#     mosaic, out_trans = merge(src_files_to_mosaic, method='first')  # 'first'来确保选择重叠的第一张影像
#
#     # Set the path for the merged output file
#     output_file = os.path.join(out_dir, variable_name)
#     print(f"Merged file saved at {output_file}")
#
#     # Write the mosaic to the output file
#     with rasterio.open(output_file, 'w', driver='GTiff', width=mosaic.shape[2], height=mosaic.shape[1], count=1,
#                        dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dst:
#         dst.write(mosaic)