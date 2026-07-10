# 将同一个文件夹下的所有_nee _gpp _reco命名的tif mosaic成一张tiff

import numpy as np
import os
import rasterio
from rasterio.merge import merge
from scipy.ndimage import generic_filter
from osgeo import gdal
from rasterio.windows import Window

# 定义一个函数，用于填充空洞值
def fill_holes(data, hole_value):
    mask = data == hole_value  # 创建空洞值的遮罩
    filled_data = data.copy()  # 创建填充后的数据副本

    # 找到非空洞值的像素并进行近邻填充
    filled_data[mask] = np.nan
    filled_data = generic_filter(filled_data, np.nanmean, size=3, mode='nearest')
    filled_data[mask] = filled_data[mask]

    return filled_data

# 给定影像的根路径
if __name__ == "__main__":
    region_base_dir = 'E:/The North America ecosystem carbon flux/Region_data/'

    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_base_dir)
    nation_path = os.path.join(region_base_dir, 'output_to_geotif')
    out_dir = os.path.join(region_base_dir, 'block_geotifs_merge_to_state')

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    state_name = state_list[0]
    state_path = os.path.join(nation_path, state_name)

    output_state_path = os.path.join(out_dir, state_name)
    if not os.path.isdir(output_state_path):
        os.makedirs(output_state_path)
    #
    output_files = ['NEE_RLM.tif', 'GPP_RLM.tif', 'RECO_RLM.tif']
    #  ************************
    # 除第0行 第0列的块， 其余块取除去 前两行，前两列的结果
    for variable_name in output_files:
        # Read various tif files corresponding to each variable
        variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]

        # Create a list of opened rasterio file objects
        src_files_to_mosaic = [rasterio.open(os.path.join(state_path, file)) for file in variable_files]

        # Merge the raster files
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Set the path for the merged output file
        output_file = os.path.join(output_state_path, variable_name)
        print(f"Merged file saved at {output_file}")

        # Write the mosaic to the output file
        with rasterio.open(output_file, 'w', driver='GTiff', width=mosaic.shape[2], height=mosaic.shape[1], count=1,
                           dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dst:
            dst.write(mosaic)

        if not any(elem in variable_name for elem in ['block_0_', '_0_' + variable_name]):
            # Crop and generate modified data for the rest of the blocks
            with rasterio.open(output_file, 'r+', driver='GTiff') as ds:
                x_size = ds.width
                y_size = ds.height
                data_array = ds.read(1, window=Window(2, 2, x_size - 2, y_size - 2))
                # Write the modified data back to the dataset
                ds.write(data_array, 1, window=Window(2, 2, x_size - 2, y_size - 2))
        print(f"The merged file is saved at {output_file}")
    #************************************
    # 相同产品之间拼接，重叠处取均值
    # for variable_name in output_files:
    #     # Read various tif files corresponding to each variable
    #     variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]
    #
    #     # Set the path for the merged output file
    #     output_file = os.path.join(output_state_path, variable_name)
    #     print(f"Merged file saved at {output_file}")
    #
    #     # Get the paths of the variable files
    #     variable_files_path = [os.path.join(state_path, file) for file in variable_files]
    #
    #     warp_options = gdal.WarpOptions(options=['-multi', '-r', 'average'], resampleAlg='average')
    #     # Execute stitching with averaging for overlapping areas
    #     gdal.Warp(output_file, variable_files_path, options=warp_options)
    # print(f"The final merged file with averaging for overlapping areas is saved at {output_file}")
    #**************************************
    #
    # for i in range(len(output_files)):
    #     variable_name = output_files[i]
    #     # Read various tif files corresponding to each variable
    #     variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]
    #
    #     # Create a list of opened rasterio file objects
    #     src_files_to_mosaic = [rasterio.open(os.path.join(state_path, file)) for file in variable_files]
    #
    #     # Merge the raster files
    #     mosaic, out_trans = merge(src_files_to_mosaic)
    #
    #     # Set the path for the original merged output file
    #     output_file_original = os.path.join(output_state_path,  'Merged_' + variable_name)
    #
    #     # Write the original merged file
    #     with rasterio.open(output_file_original, 'w', driver='GTiff', width=mosaic.shape[2], height=mosaic.shape[1],
    #                        count=1,
    #                        dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dst:
    #         dst.write(mosaic)
    #     print(f"The original merged file is saved at {output_file_original}")
    #
    #     with rasterio.open(output_file_original, 'w') as ds:
    #         data = ds.read(1)  # 读取数据
    #         filled_data = fill_holes(data, files_and_hole_values[i])  # 使用定义的函数填充空洞值
    #         # 将填充后的数据写回到新数据集，并在保存时在文件名前加上"merged_"
    #         output_file_filled = os.path.join(output_state_path, 'merged_' + variable_name)
    #         ds.write(filled_data, 1)
    #     print(f"The filled merged file is saved at {output_file_filled}")
    #
    #     # # Check condition and write to a new output file
    #     if not any(elem in variable_name for elem in ['block_0_', '_0_' + variable_name]):
    #         # Create a new output file path for the modified data
    #         output_file_modified = os.path.join(output_state_path, 'Modified_' + variable_name)
    #
    #         # Crop and generate modified data for the rest of the blocks
    #         with rasterio.open(output_file_original, 'r') as ds_original, rasterio.open(output_file_modified, 'w',
    #                                                                                     driver='GTiff',
    #                                                                                     width=mosaic.shape[2],
    #                                                                                     height=mosaic.shape[1], count=1,
    #                                                                                     dtype=mosaic.dtype,
    #                                                                                     crs=src_files_to_mosaic[0].crs,
    #                                                                                     transform=out_trans) as ds_modified:
    #             x_size = ds_original.width
    #             y_size = ds_original.height
    #             data_array = ds_original.read(1, window=Window(2, 2, x_size, y_size))
    #             # Write the modified data to the new output file
    #             ds_modified.write(data_array, 1, window=Window(2, 2, x_size, y_size))
    #
    #         print(f"The modified file is saved at {output_file_modified}")
    #
    #     print(f"The original merged file is saved at {output_file_original}")

        # print(f"The merged file is saved at {output_file}")
    # output_files = ['NEE_RLM2.tif', 'GPP_RLM2.tif', 'RECO_RLM2.tif']
    # block_size = 32  # Block size is 32x32 pixels
    #
    # for variable_name in output_files:
    #     # Read various tif files corresponding to each variable
    #     variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]
    #
    #     # Set the path for the merged output file
    #     output_file = os.path.join(output_state_path, 'SM_Merged_sg_' + variable_name)
    #     print(f"Merged file saved at {output_file}")
    #
    #     # Get the paths of the variable files
    #     variable_files_path = [os.path.join(state_path, file) for file in variable_files]
    #
    #     warp_options = gdal.WarpOptions(options=['-multi', '-r', 'average'], resampleAlg='max')
    #     # Execute stitching with averaging for overlapping areas
    #     gdal.Warp(output_file, variable_files_path, options=warp_options)
    #
    #     if not any(elem in variable_name for elem in ['block_0_', '_0_' + variable_name]):
    #         # Crop and generate modified data for the rest of the blocks
    #         ds = gdal.Open(output_file, gdal.GA_Update)
    #         x_size = ds.RasterXSize
    #         y_size = ds.RasterYSize
    #         data_array = ds.GetRasterBand(1).ReadAsArray()
    #
    #     print(f"The merged file is saved at {output_file}")
    #     # Apply mean smoothing to the entire image
    #     ds = gdal.Open(output_file, gdal.GA_Update)
    #     band = ds.GetRasterBand(1)
    #
    #     # Define the mean kernel (3x3)
    #     kernel = np.ones((3, 3), dtype=float) / 9
    #
    #     # Apply the mean filter
    #     smoothed_data = band.ReadAsArray(xoff=1, yoff=1, win_xsize=x_size - 2, win_ysize=y_size - 2)
    #     smoothed_data = signal.convolve2d(smoothed_data, kernel, mode='same')
    #     band.WriteArray(smoothed_data, xoff=1, yoff=1)
    #
    #     del ds, band  # Close the dataset after smoothing
    #     print(f"The merged file with mean smoothing is saved at {output_file}")


    # output_files = ['NEE_RLM2.tif', 'GPP_RLM2.tif', 'RECO_RLM2.tif']
    # for variable_name in output_files:
    #     # Read various tif files corresponding to each variable
    #     variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]
    #
    #     # Set the path for the merged output file
    #     output_file = os.path.join(output_state_path, 'SM_Merged_av_'+variable_name)
    #     print(f"Merged file saved at {output_file}")
    #
    #     # Get the paths of the variable files
    #     variable_files_path = [os.path.join(state_path, file) for file in variable_files]
    #
    #     warp_options = gdal.WarpOptions(options=['-multi', '-r', 'average'], resampleAlg='max')
    #     # Execute stitching with averaging for overlapping areas
    #     gdal.Warp(output_file, variable_files_path, options=warp_options)
    #
    #     if not any(elem in variable_name for elem in ['block_0_', '_0_'+variable_name]):
    #         # Crop and generate modified data for the rest of the blocks
    #         ds = gdal.Open(output_file)
    #         x_size = ds.RasterXSize
    #         y_size = ds.RasterYSize
    #         ds.GetRasterBand(1).WriteArray(ds.GetRasterBand(1).ReadAsArray()[2:x_size - 2, 2:y_size - 2])
    #         del ds
    #
    #     print(f"The merged file is saved at {output_file}")

        # # Apply bilateral filtering to the entire image
        # ds = gdal.Open(output_file, gdal.GA_Update)
        # band = ds.GetRasterBand(1)
        #
        # # Apply the bilateral filter
        # smoothed_data = band.ReadAsArray(xoff=1, yoff=1, win_xsize=x_size - 2, win_ysize=y_size - 2)
        # bilateral_smoothed_data = gaussian_filter(smoothed_data, sigma=1.5, mode='reflect')
        #
        # band.WriteArray(bilateral_smoothed_data, xoff=1, yoff=1)
        # del ds, band  # Close the dataset after smoothing
        # print(f"The merged file with bilateral filtering is saved at {output_file}")

        # # Apply median filtering to the entire image
        # ds = gdal.Open(output_file, gdal.GA_Update)
        # band = ds.GetRasterBand(1)
        #
        # # Define the window size for the median filter
        # window_size = (3, 3)  # Adjust the window size as needed
        #
        # # Apply the median filter
        # smoothed_data = band.ReadAsArray(xoff=1, yoff=1, win_xsize=x_size - 2, win_ysize=y_size - 2)
        # rows, cols = smoothed_data.shape
        # for i in range(1, rows - 1):
        #     for j in range(1, cols - 1):
        #         window = band.ReadAsArray(xoff=j - 1, yoff=i - 1, win_xsize=3, win_ysize=3)
        #         median_value = np.median(window)
        #         smoothed_data[i, j] = median_value
        #
        # band.WriteArray(smoothed_data, xoff=1, yoff=1)
        # del ds, band  # Close the dataset after smoothing
        # print(f"The merged file with median filtering is saved at {output_file}")
        # # Apply mean smoothing to the entire image
        # ds = gdal.Open(output_file, gdal.GA_Update)
        # band = ds.GetRasterBand(1)
        #
        # # Define the mean kernel (3x3)
        # kernel = np.ones((3, 3), dtype=float) / 9
        #
        # # Apply the mean filter
        # smoothed_data = band.ReadAsArray(xoff=1, yoff=1, win_xsize=x_size - 2, win_ysize=y_size - 2)
        # smoothed_data = signal.convolve2d(smoothed_data, kernel, mode='same')
        # band.WriteArray(smoothed_data, xoff=1, yoff=1)
        #
        # del ds, band  # Close the dataset after smoothing
        # print(f"The merged file with mean smoothing is saved at {output_file}")

        # # Smooth only the overlapping areas between blocks
        # ds = gdal.Open(output_file)
        # x_size = ds.RasterXSize
        # y_size = ds.RasterYSize
        #
        # # Smooth the overlapping areas of blocks
        # for i in range(len(variable_files_path) - 1):
        #     current_ds = gdal.Open(variable_files_path[i])
        #     next_ds = gdal.Open(variable_files_path[i + 1])
        #
        #     # Extract overlapping areas for smoothing (assuming 2-pixel overlap)
        #     overlap_region_current = current_ds.GetRasterBand(1).ReadAsArray()[:, -2:]
        #     overlap_region_next = next_ds.GetRasterBand(1).ReadAsArray()[:, :2]
        #
        #     # Smooth the overlapping regions
        #     ds.GetRasterBand(1).WriteArray(ds.GetRasterBand(1).ReadAsArray())
        #     ds.GetRasterBand(1).WriteArray(overlap_region_current, x_size - 2, y_size)
        #     ds.GetRasterBand(1).WriteArray(overlap_region_next, 0, y_size)
        #
        #     current_ds = None
        #     next_ds = None
        #
        # del ds
        # print(f"The merged file with smoothed overlapping areas is saved at {output_file}")

    # # 进入州下面的block目录
    # output_files = ['NEE_RLM2.tif', 'GPP_RLM2.tif', 'RECO_RLM2.tif']
    # for variable_name in output_files:
    #     # 读取各个变量对应的tif文件
    #     variable_files = [file for file in os.listdir(state_path) if file.endswith(variable_name)]
    #     # 将这些variable_files镶嵌成一个完整的tif
    #     # 设置合并后的输出文件名
    #     output_file = os.path.join(output_state_path, 'Merged_' + variable_name)
    #     print(f"Merged file saved at {output_file}")
    #
    #     # 设置合并后的输出文件名
    #     output_file = os.path.join(output_state_path, 'Merged_' + variable_name)
    #
    #     # 获取变量文件的路径
    #     variable_files_path = [os.path.join(state_path, file) for file in variable_files]
    #     warp_options = gdal.WarpOptions(options=['-multi', '-r', 'average'])
    #     # 执行拼接
    #     # gdal.Warp(output_file, variable_files_path,options=['-multi'])
    #     gdal.Warp(output_file, variable_files_path, options=warp_options)
    #
    #     print(f"合并后的文件保存在 {output_file}")

        # # Open the merged TIF file
        # merged_dataset = gdal.Open(output_file, gdal.GA_ReadOnly)
        #
        # # Read the merged data as a numpy array
        # merged_data = merged_dataset.ReadAsArray()
        #
        # # Apply mean filtering to the merged data
        # smoothed_data = median_filter(merged_data, size=3,
        #                                mode='constant')  # Adjust the size parameter as needed for the desired smoothing effect
        #
        # # Define the output path for the smoothed TIF file
        # smoothed_output_file = os.path.join(output_state_path, 'Smoothed_Merged_' + variable_name)
        #
        # # Create a new TIF file and write the smoothed data
        # driver = gdal.GetDriverByName('GTiff')
        # smoothed_dataset = driver.Create(smoothed_output_file, merged_dataset.RasterXSize, merged_dataset.RasterYSize,
        #                                  1, gdal.GDT_Float32)
        #
        # smoothed_dataset.SetGeoTransform(merged_dataset.GetGeoTransform())
        # smoothed_dataset.SetProjection(merged_dataset.GetProjection())
        # smoothed_dataset.GetRasterBand(1).WriteArray(smoothed_data)
        #
        # # Close the datasets
        # merged_dataset = None
        # smoothed_dataset = None
        #
        # print(f"The smoothed result is saved at {smoothed_output_file}")
        # # # 开第一个文件来获取其参数
        # dataset = gdal.Open(variable_files_path[0], gdalconst.GA_ReadOnly)
        #
        # # 获取数据的驱动和波段数
        # driver = dataset.GetDriver()
        # band = dataset.GetRasterBand(1)
        #
        # # 创建一个与输入文件相同大小和数据类型的新文件
        # merged_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize,
        #                                len(variable_files_path), band.DataType)
        #
        # # 将其他文件内容写入新文件
        # for i, file in enumerate(variable_files_path):
        #     dataset = gdal.Open(file, gdal.GA_ReadOnly)
        #     merged_dataset.GetRasterBand(i + 1).WriteArray(dataset.GetRasterBand(1).ReadAsArray())
        #
        # # 清理和关闭数据集
        # dataset = None
        # merged_dataset = None




