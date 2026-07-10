# 将下载的同一区域的ref fpar/lai meteorology 三个产品分成32*32像素大小的块输出保存

import os
import numpy as np
import rasterio
from numpy.lib.stride_tricks import as_strided
from rasterio.windows import Window
from rasterio.transform import from_origin

def sliding_window_view(arr, window_shape):
    arr = np.ascontiguousarray(arr)
    # Get the shape and strides of the input array
    shape = np.array(arr.shape * 2)
    strides = np.array(arr.strides * 2)

    # Set the shape of the sliding window view
    shape[:arr.ndim] -= np.array(window_shape - 1)
    shape[arr.ndim:] = window_shape

    # Set the strides of the sliding window view
    strides[:arr.ndim] += arr.strides
    strides[arr.ndim:] = arr.strides

    return as_strided(arr, shape=shape, strides=strides)

def read_multiband(ref_doy_path):
    # 读取并打开文件
    # Open the multi-band TIFF file
    with rasterio.open(ref_doy_path) as src:
        # Read all bands into a single numpy array
        data = src.read()

    # Get the number of bands and the dimensions of the array
    num_bands, height, width = data.shape

    # Define the window size
    window_size = 3

    # Calculate the shape of the output array
    output_shape = (height, width, num_bands, window_size, window_size)

    # Reshape the array to create the sliding window view
    sub_arrays = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=data.itemsize * np.array(
        [width * num_bands, num_bands, width * num_bands, num_bands, 1]))
    # The sub_arrays variable now contains the data arranged in the required shape
    print(sub_arrays.shape)
    return sub_arrays,height, width

def read_and_process_tiff_blocks(tiff_path, block_size=32):
    with rasterio.open(tiff_path) as src:
        width = src.width
        height = src.height

        def generate_blocks():
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    window = rasterio.windows.Window(x, y, min(block_size, width - x), min(block_size, height - y))
                    data = src.read(window=window)
                    yield x, y, data

        for x, y, data in generate_blocks():
            print(f"Block at (x={x}, y={y}) - Shape: {data.shape}")
            # 在这里可以进行进一步处理，例如保存到新文件或进行其他计算/可视化操作

def read_in_blocks_with_specific_overlap(file_path, state_output_path, product_name, date, geoinfo_base_dir, state_name):
    # Define the block size and overlap
    block_size = 32
    overlap = 2
    with rasterio.open(file_path) as src:
        for i in range(0, src.height, block_size - overlap):
            for j in range(0, src.width, block_size - overlap):
                # Determine the block indices
                block_index_i = i // (block_size - overlap)
                block_index_j = j // (block_size - overlap)
                # Create the directory for the current block
                block_output_dir = os.path.join(state_output_path, f'block_{block_index_i}_{block_index_j}')
                if not os.path.isdir(block_output_dir):
                    os.makedirs(block_output_dir)
                # Define the output block file path
                block_output_path = os.path.join(block_output_dir, f'{product_name}_{date}.tif')
                # Define the window with overlap
                window = Window(j, i, block_size, block_size)
                # Read the data block from the source
                data_block = src.read(window=window)
                # Write the block to the output path
                with rasterio.open(block_output_path, 'w', driver='GTiff', height=block_size, width=block_size,
                                   count=src.count, dtype=src.dtypes[0], crs=src.crs,
                                   transform=from_origin(src.bounds.left + j * src.res[0],
                                                         src.bounds.top - i * src.res[1],
                                                         src.res[0], src.res[1])) as dst:
                    dst.write(data_block)
                geotransformation_dir = os.path.join(geoinfo_base_dir, state_name)
                if not os.path.isdir(geotransformation_dir):
                    os.makedirs(geotransformation_dir)
                geotransformation_block_dir = os.path.join(geotransformation_dir, f'block_{block_index_i}_{block_index_j}')
                if not os.path.isdir(geotransformation_block_dir):
                    os.makedirs(geotransformation_block_dir)
                # 获取待输出影像的Origin和Pixel Size参数
                output_origin = src.bounds.left + j * src.res[0], src.bounds.top - i * src.res[1]
                output_pixel_size = src.res[0], src.res[1]
                geotransformation_file = os.path.join(geotransformation_block_dir,
                                                      f'block_{block_index_i}_{block_index_j}_geoinfo.txt')
                with open(geotransformation_file, 'w') as f:
                    f.write(f'Output Origin: {output_origin}\n')
                    f.write(f'Output Pixel Size: {output_pixel_size}\n')


if __name__ == '__main__':
    # 所有region 存储的根目录 每个国家的文件夹
    region_base_dir =  "/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/"
    region_dir = os.path.join(region_base_dir,'original_images')
    out_base_dir = os.path.join(region_base_dir,'images_split_to_blocks')
    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_dir)
    nation_path = os.path.join(region_dir, 'gadm41_USA_1')

    geoinfo_base_dirname = 'blocks_to_input_npy'
    geoinfo_base_dir = os.path.join(region_base_dir, geoinfo_base_dirname)
    if not os.path.isdir(geoinfo_base_dir):
        os.makedirs(geoinfo_base_dir)

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    for s in range(31,40): # len(state_list)
        state_name = state_list[s]
        # If the output directory for the product does not exist, create it
        state_output_path = os.path.join(out_base_dir, state_name)
        if not os.path.isdir(state_output_path):
            os.makedirs(state_output_path)
        state_path = os.path.join(nation_path,state_name)
        # 进入每个州下面的产品目录
        # product_list = os.listdir(state_path)
        era5land_path = os.path.join(state_path,'DAILY_AGGR')
        ref_path = os.path.join(state_path, 'MCD43A4')
        lai_path = os.path.join(state_path, 'MCD15A3H')
        # # Iterate through each product directory
        # product_directories = [ref_path, lai_path,era5land_path]
        product_directories = [ref_path, lai_path,era5land_path] # ref_path,era5land_path
        for directory in product_directories:
            product_name = os.path.basename(directory)
            # Iterate through each file in the directory
            # for file in os.listdir(directory):
            filelist = os.listdir(directory)
            for f in range(0,len(os.listdir(directory))):# len(os.listdir(directory))
                file = filelist[f]
                file_path = os.path.join(directory, file)
                date = file.split('.')[0]
                read_in_blocks_with_specific_overlap(file_path, state_output_path,product_name,date,geoinfo_base_dir,state_name)
                print(file)

    # # 统一路径中的斜杠
    # ref_image_path = ref_path.replace('\\', '/')
    # lai_image_path = lai_path.replace('\\', '/')
    # mete_image_path = era5land_path.replace('\\', '/')
    # ref_year_list = []
    # lai_year_list = []
    # mete_year_list = []
    # ref_doy_list = os.listdir(ref_image_path)
    # for d in range(0, 5):
    #     ref_doy_path = os.path.join(ref_image_path, ref_doy_list[d])
    #     # 分块读取tif文件
    #     read_and_process_tiff_blocks(ref_doy_path, block_size=32)
    #
    #     ref_datasets,ref_height, ref_width = read_multiband(ref_doy_path)
    #     ref_year_list.append(ref_datasets)
    #     # fpar/lai影像
    #     ref_date_filename = ref_doy_list[d]
    #     lai_doy_path = os.path.join(lai_image_path, ref_date_filename)
    #     # lai_doy_path = lai_doy_path.replace('\\', '/')
    #     if not os.path.isdir(lai_doy_path):
    #         lai_datasets = np.full((ref_height, ref_width,4, 3, 3), -9999)
    #         lai_year_list.append(lai_datasets)
    #     else:
    #         lai_datasets,lai_height, lai_width = read_multiband(lai_doy_path)
    #         if lai_datasets is None:
    #             lai_datasets = np.full((ref_height, ref_width, 4, 3, 3), -9999)
    #             lai_year_list.append(lai_datasets)
    #         else:
    #             lai_year_list.append(lai_datasets)
    #
    #     # mete影像
    #     mete_doy_path = os.path.join(mete_image_path, ref_date_filename)
    #     # mete_doy_path = mete_doy_path.replace('\\', '/')
    #     if not os.path.isdir(mete_doy_path):
    #         mete_datasets = np.full((ref_height, ref_width, 6, 3, 3), -9999)
    #         mete_year_list.append(mete_datasets)
    #     else:
    #         mete_datasets, mete_height, mete_width = read_multiband(mete_doy_path)
    #         if mete_datasets is None:
    #             mete_datasets = np.full((ref_height, ref_width, 6, 3, 3), -9999)
    #             mete_year_list.append(mete_datasets)
    #         else:
    #             mete_year_list.append(mete_datasets)
    #     mete_year_list
    #     lai_year_list
    #     ref_year_list
    #     print('hhhh')





