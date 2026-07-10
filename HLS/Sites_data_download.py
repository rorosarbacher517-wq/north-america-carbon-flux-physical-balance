
import os
import earthaccess
import rioxarray as rxr
import numpy as np
import pandas as pd
import requests
import rioxarray
from rasterio.transform import from_origin
from rasterio.windows import Window
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import geopandas as gp
from osgeo import gdal
from skimage import io
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
import hvplot.xarray
import hvplot.pandas
import earthaccess
from rasterio.io import MemoryFile
import rasterio
#
# # There are a couple plots that generate errors that we want to ignore.
import warnings
warnings.filterwarnings('ignore')
#
auth = earthaccess.login(strategy="netrc")
auth = earthaccess.login(persist=True)
# Authenticate using Earthaccess
# earthaccess.login(persist=True)
#
stations_file = "E:/HLS carbon flux/data/sites_263_shp/sites_263_1.5km_buffer.geojson"  # 替换为实际的 CSV 文件路径
stations = gp.read_file(stations_file)

# 定义输出目录
output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/'
os.makedirs(output_dir, exist_ok=True)

# 定义年份范围
years = range(2015, 2016)


def download_hls_data(bounding_box, year):
    granules = earthaccess.search_data(
        short_name=['HLSL30', 'HLSS30'],
        bounding_box=bounding_box,  # 使用提供的边界框
        temporal=(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"),
    )
    return granules


# 提取波段名称和产品名
def extract_band_and_product_name(url):
    base_name = os.path.basename(url)
    parts = base_name.split('.')
    product_name = ''.join(parts[1:-4])  # 提取产品名
    return product_name


def download_file_with_retry(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # 如果请求返回码不是200，将引发HTTPError
            return response.content
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)  # 等待一段时间后重试
    return None  # 如果所有重试都失败


# 提取所需要的波段
def extract_bands(product_links):
    serface_band_links = []
    # print(product_links[0].split('/')[4])
    # Define which HLS product is being accessed
    if product_links[0].split('/')[4] == 'HLSS30.020':
        serface_bands = ['B01', 'B02', 'B03', 'B04', 'B8A', 'B11', 'B12',
                         'Fmask']  # Coastal/Aerosol Blue Green Red NIR Narrow SWIR1 SWIR2 for L30
    else:
        serface_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                         'Fmask']  # Coastal/Aerosol Blue Green Red NIR Narrow SWIR1 SWIR2 for L30

    # Subset the assets in the item down to only the desired bands
    for a in product_links:
        if any(b in a for b in serface_bands):
            serface_band_links.append(a)
    return serface_band_links


import rioxarray as rxr  # 确保你导入了正确的库


# def scale_and_mask_old(subbands_links):
#     chunk_size = dict(band=1, x=512, y=512)  # Tiles have 1 band and are divided into 512x512 pixel chunks

#     # 处理每个波段链接
#     for e in subbands_links:
#         print(e)
#         # 获取波段类型
#         # band_type = e.split('/')[-1]
#         # print(band_type)
#         band_type = e.rsplit('.', 2)[-2]  # 提取波段名称部分
#         print(band_type)
#         # 开放并构建数据集
#         if band_type in ['B03', 'B01', 'B04', 'B06', 'B05', 'B07', 'B02', 'B8A', 'B11', 'B12']:  # 识别需要缩放的波段
#             band_data = rxr.open_rasterio(e, chunks=chunk_size, masked=True).squeeze('band', drop=True)
#             band_data.attrs['scale_factor'] = 0.0001  # 设定缩放因子

#             # 可选：如果需要处理数据，比如存储或进一步分析，可以在这里添加代码
#             print(f"{band_type} band scaled with factor {band_data.attrs['scale_factor']}")

#         elif band_type == 'Fmask':  # 检查是否为 Fmask 波段
#             fmask_data = rxr.open_rasterio(e, chunks=chunk_size, masked=True).squeeze('band', drop=True)
#             # 不需要缩放，因此在这里可以直接处理 Fmask 数据，或者继续数据处理

#             print(f"Fmask band loaded without scaling")
#     print("The COGs have been loaded into memory!")


def scale_and_mask(subbands_links):
    chunk_size = dict(band=1, x=512, y=512)  # Tiles have 1 band and are divided into 512x512 pixel chunks

    # 处理每个波段链接
    for e in subbands_links:
        print(e)

        # 获取波段类型
        band_type = e.rsplit('.', 2)[-2]  # 提取波段名称部分
        print(band_type)

        # 开放并构建数据集
        try:
            if band_type in ['B03', 'B01', 'B04', 'B06', 'B05', 'B07', 'B02', 'B8A', 'B11', 'B12']:  # 识别需要缩放的波段
                # 下载文件
                local_filename = 'local_' + band_type + '.tif'
                r = requests.get(e)
                with open(local_filename, 'wb') as f:
                    f.write(r.content)

                # 使用 rasterio 打开下载后的文件
                with rasterio.open(local_filename) as src:
                    band_data = src.read(1)  # 读取第一波段

                    # 获取坐标
                    height, width = band_data.shape
                    x = np.arange(src.transform[2], src.transform[2] + width * src.transform[0], src.transform[0])
                    y = np.arange(src.transform[5], src.transform[5] + height * src.transform[4], src.transform[4])

                    # 创建 xarray 数据数组
                    da = xr.DataArray(
                        band_data,
                        dims=["y", "x"],
                        coords={"y": y, "x": x},
                    )

                    da.attrs['scale_factor'] = 0.0001  # 设定缩放因子
                    print(f"{band_type} band scaled with factor {da.attrs['scale_factor']}")

                # 可选：在这里可以处理或存储数据

                os.remove(local_filename)  # 清理本地文件

            elif band_type == 'Fmask':  # 检查是否为 Fmask 波段
                # 下载文件
                local_filename = 'local_Fmask.tif'
                r = requests.get(e)
                with open(local_filename, 'wb') as f:
                    f.write(r.content)

                # 使用 rasterio 打开下载后的文件
                with rasterio.open(local_filename) as src:
                    fmask_data = src.read(1)  # 读取第一波段

                    # 获取坐标
                    height, width = fmask_data.shape
                    x = np.arange(src.transform[2], src.transform[2] + width * src.transform[0], src.transform[0])
                    y = np.arange(src.transform[5], src.transform[5] + height * src.transform[4], src.transform[4])

                    # 创建 xarray 数据数组
                    fmask_da = xr.DataArray(
                        fmask_data,
                        dims=["y", "x"],
                        coords={"y": y, "x": x},
                    )

                    print(f"Fmask band loaded without scaling")

                os.remove(local_filename)  # 清理本地文件

        except Exception as ex:
            print(f"Error occurred while processing {band_type}: {ex}")

    print("The COGs have been loaded into memory!")


def proj_transfer(subbands_link):
    # 下载文件
    local_filename = 'local_raster.tif'
    try:
        r = requests.get(subbands_link)
        with open(local_filename, 'wb') as f:
            f.write(r.content)

        # 使用 rasterio 打开下载后的文件
        with rasterio.open(local_filename) as src:
            # 获取 CRS (Coordinate Reference System) 作为 WKT
            fsUTM = src.crs.to_wkt()
            # fsUTM = src.spatial_ref.crs_wkt
    except Exception as ex:
        print(f"Error occurred while processing the raster: {ex}")
        fsUTM = None  # 在发生错误时返回 None

    finally:
        # 清理本地文件
        if os.path.exists(local_filename):
            os.remove(local_filename)

    return fsUTM


# # Define function to scale
# def scaling(band):
#     scale_factor = band.attrs['scale_factor']
#     band_out = band.copy()
#     band_out.data = band.data*scale_factor
#     band_out.attrs['scale_factor'] = 1
#     return(band_out)
def scaling(band):
    """
    Scale the band data by the scale factor,
    skipping nodata values which are defined by _FillValue (-9999).

    Parameters:
    - band: xarray DataArray which contains the band data and attributes.

    Returns:
    - scaled_band: xarray DataArray with scaled data.
    """
    scale_factor = band.attrs['scale_factor']  # Get the scale factor
    band_out = band.copy()  # Create a copy for output

    # Create a mask for the nodata values
    nodata_mask = (band_out.data == -9999)

    # Scale the data while preserving nodata values
    band_out.data = np.where(nodata_mask, -9999, band_out.data * scale_factor)

    # Update the scale factor attribute
    band_out.attrs['scale_factor'] = 1

    return band_out


import requests
import rasterio
import xarray as xr
import os
import os
import geopandas as gpd
import rasterio
import rioxarray as rxr


def download_file(subband_url, username, password):
    """Download a file from the given URL and return the local filename."""
    # Specify the local filename
    local_filename = "/content/HLS.L30.T16UFD.2015006T162610.v2.0.B03.tif"

    response = requests.get(subband_url, auth=(username, password))

    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {local_filename}")
        return local_filename
    else:
        print(f"Failed to download file from {subband_url}: {response.status_code}")
        return None


# def process_subbands(subbands_links, fsUTM, station, username, password):

#     # Convert the single station geometry to a GeoDataFrame for CRS transformation

#     # station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=station.crs)

#     # Specify the local filename
#     processed_bands = []  # List to hold processed band data

#     for subband in subbands_links:

#         print(f"Processing subband: {subband}")

#         band_type = subband.rsplit('.', 2)[-2]  # 提取波段名称部分

#         print(subband)

#         # Download the file and get the local filename

#         local_filename = download_file(subband, username, password)

#         # local_filename = download_file(subband)

#         print(local_filename)

#         if local_filename is None:

#             continue  # Skip if download failed

#         # Open the raster file using Rasterio

#         try:

#             with rasterio.open(local_filename) as src:

#                 # Get the CRS of the raster

#                 raster_crs = src.crs

#                 # Transform the station's geometry to the CRS of the raster

#                 first_station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=stations.crs)

#                 first_station_transformed = first_station_gdf.to_crs(raster_crs)

#                 # Load the raster data

#                 subband_data = rxr.open_rasterio(local_filename)

#                 # Clip the raster based on the transformed first station geometry

#                 cropped_data = subband_data.rio.clip(first_station_transformed.geometry.values, first_station_transformed.crs, all_touched=True)
#             # Scale the data if it's not 'Fmask'
#                 if band_type != 'Fmask':
#                     subband_cropped_scaled = scaling(cropped_data)
#                     print(subband_cropped_scaled.shape)

#                     processed_bands.append(subband_cropped_scaled)  # Add to processed list
#                     print(f"Scaled data for {band_type}")
#                 else:
#                     processed_bands.append(cropped_data)  # Append raw cropped data for Fmask
#                     print(cropped_data.shape)
#                     print(f"'Fmask' data detected for {band_type}, skipping scaling.")

#                 # Optional: Process the cropped data further
#                 print(subband_cropped_scaled)

#         except Exception as e:
#             print(f"An error occurred: {e}")

#         # Clean up the local file if needed
#         if os.path.exists(local_filename):
#             os.remove(local_filename)
#             print(f"Removed local file: {local_filename}")

#     return processed_bands  # Return the list of processed bands

# def process_subbands(subbands_links, fsUTM, station, username, password,station_year_product_folder):
#     # Convert the single station geometry to a GeoDataFrame for CRS transformation
#     # station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=station.crs)
#     processed_bands = []  # List to hold processed band data

#     for subband in subbands_links:
#         print(f"Processing subband: {subband}")
#         file_name = subband.split('/')[-1]
#         file_path = os.path.join(station_year_product_folder, file_name)
#         band_type = subband.rsplit('.', 2)[-2]  # Extract band name

#         # Download the file and get the local filename
#         local_filename = download_file(subband, username, password)
#         if local_filename is None:
#             continue  # Skip if download failed

#         try:
#             with rasterio.open(local_filename) as src:
#                 # Get the CRS of the raster
#                 raster_crs = src.crs

#                 # Transform the station's geometry to the CRS of the raster
#                 first_station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=stations.crs)

#                 first_station_transformed = first_station_gdf.to_crs(raster_crs)
#                 # first_station_transformed = first_station_gdf.to_crs(fsUTM)
#                 # Load the raster data
#                 subband_data = rxr.open_rasterio(local_filename)

#                 # Clip the raster based on the transformed station geometry
#                 cropped_data = subband_data.rio.clip(first_station_transformed.geometry.values,
#                                                       first_station_transformed.crs, all_touched=True)
#                 # 打印cropped_data的shape
#                 print(cropped_data.shape)
#                 print("Cropped Data CRS:", cropped_data.rio.crs)  # Should show the CRS
#                 print("Cropped Data Transform:", cropped_data.rio.transform())  # Shows the affine transformation

#                 # Scale the data if it's not 'Fmask'
#                 if band_type != 'Fmask':
#                     subband_cropped_scaled = scaling(cropped_data)
#                     processed_bands.append(subband_cropped_scaled)  # Add scaled data
#                     print(f"Scaled data for {band_type}")
#                 else:
#                     subband_cropped_scaled = cropped_data
#                     processed_bands.append(subband_cropped_scaled)  # Append raw Fmask data
#                     print(f"'Fmask' data detected for {band_type}, skipping scaling.")

#                 # 将cropped_data保存成tif 投影统一成4326，transfer就用cropped_data.rio.transform()
#                 with open(file_path, 'wb') as f:
#                    f.write(subband_cropped_scaled)
#                   print(f'Successfully downloaded: {file_name}')

#         except Exception as e:
#             print(f"An error occurred: {e}")

#         # Clean up the local file if needed
#         if os.path.exists(local_filename):
#             os.remove(local_filename)
#             print(f"Removed local file: {local_filename}")

#     # 将processed_bands转为array,打印shape 然后reshape为(band, height, width)的形式

#     # Export the processed bands to a multi-band GeoTIFF
#     if processed_bands:
#          # Stack the processed bands into a single array
#         stacked_array = np.stack([band.data for band in processed_bands])

#         # Print the shape before reshaping
#         print(f"Shape of stacked array before reshaping: {stacked_array.shape}")

#         # Reshape to (band, height, width)
#         # Ensure the dimensions align correctly
#         reshaped_array = stacked_array.reshape(len(processed_bands), *stacked_array.shape[2:])
#         export_masked_data_as_tif(reshaped_array, raster_crs, output_filename= output_filename+".tif")
#         print(f"Shape of reshaped array: {reshaped_array.shape}")

#     return processed_bands  # Return the list of processed bands
def process_subbands(subbands_links, fsUTM, station, username, password, station_year_product_folder):
    processed_bands = []  # List to hold processed band data

    for subband in subbands_links:
        print(f"Processing subband: {subband}")

        # Extract the file name and set the file path
        file_name = subband.split('/')[-1]
        print(file_name)
        file_path = os.path.join(station_year_product_folder, file_name)

        band_type = subband.rsplit('.', 2)[-2]  # Extract band name

        # Download the file and get the local filename
        local_filename = download_file(subband, username, password)

        if local_filename is None:
            print(f"Failed to download file for {subband}")
            continue  # Skip if download failed

        try:
            with rasterio.open(local_filename) as src:
                # Get the CRS of the raster
                raster_crs = src.crs

                # Transform the station's geometry to the CRS of the raster
                first_station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=stations.crs)
                first_station_transformed = first_station_gdf.to_crs(raster_crs)

                # Load the raster data
                subband_data = rxr.open_rasterio(local_filename)

                # Clip the raster based on the transformed station geometry
                cropped_data = subband_data.rio.clip(
                    first_station_transformed.geometry.values,
                    first_station_transformed.crs,
                    all_touched=True
                )

                # Check for empty cropped_data
                if cropped_data.size == 0:
                    print(f"No data found in cropped_data for {band_type}")
                    continue

                # Print cropped_data's shape and metadata
                print(cropped_data.shape)
                print("Cropped Data CRS:", cropped_data.rio.crs)
                print("Cropped Data Transform:", cropped_data.rio.transform())

                # Scale the data if it's not 'Fmask'
                if band_type != 'Fmask':
                    subband_cropped_scaled = scaling(cropped_data)
                else:
                    subband_cropped_scaled = cropped_data

                processed_bands.append(subband_cropped_scaled)  # Add processed data
                print(f"Processed data for {band_type}")

                # Define target CRS and transform for saving
                target_crs = 'EPSG:3857'
                target_transform = rasterio.transform.from_origin(
                    cropped_data.rio.bounds()[0],  # minx
                    cropped_data.rio.bounds()[3],  # maxy
                    abs(cropped_data.rio.res[0]),  # pixel width
                    abs(cropped_data.rio.res[1])  # pixel height (should be negative)
                )

                # Save the cropped data to a GeoTIFF
                with rasterio.open(
                        file_path,
                        'w',
                        driver='GTiff',
                        height=cropped_data.shape[1],
                        width=cropped_data.shape[2],
                        count=1,  # Assuming single band after clipping
                        dtype=cropped_data.dtype,
                        crs=target_crs,
                        transform=target_transform
                ) as dst:
                    dst.write(cropped_data)
                # with open(file_path, 'wb') as f:
                #     f.write(subband_cropped_scaled)

                print(f'Successfully saved: {file_path}')

        except Exception as e:
            print(f"An error occurred during processing of {subband}: {e}")


def export_masked_data_as_tif(reshaped_array, raster_crs, output_filename="output.tif"):
    """
    Export the processed bands as a multi-band GeoTIFF.

    Parameters:
    - reshaped_array: Array representing multiple bands with shape (band, height, width)
    - raster_crs: Coordinate reference system of the raster.
    - output_filename: Name of the output file.
    """

    # Get the number of bands, height, and width from the reshaped_array
    num_bands = reshaped_array.shape[0]
    height, width = reshaped_array.shape[1:]  # Assuming the shape is (band, height, width)

    # Define metadata for the GeoTIFF file
    metadata = {
        'driver': 'GTiff',
        'count': num_bands,  # Number of bands
        'height': height,
        'width': width,
        'dtype': 'float32',  # This should be adjusted based on your data type
        'crs': raster_crs,  # Use CRS obtained from the source raster
        'transform': transform  # Modify as necessary

    }

    # Write the data to the GeoTIFF file
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        for i in range(num_bands):
            dst.write(reshaped_array[i], i + 1)  # Write each band, index starts at 1

    print(f"Exported processed data to {output_filename}")


# # Quality Filtering
# def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
#     """
#     Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
#     By default, bits 1-5 are used.
#     """
#     mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
#     # Remove/Mask Fill Values and Convert to Integer
#     quality_data = np.nan_to_num(quality_data, 0).astype(np.int8)
#     for bit in bit_nums:
#         # Create a Single Binary Mask Layer
#         mask_temp = np.array(quality_data) & 1 << bit > 0
#         mask_array = np.logical_or(mask_array, mask_temp)
#     return mask_array
def create_quality_mask(quality_data, bit_nums):
    """
    Create a binary mask layer based on quality data.

    Parameters:
    - quality_data: input data with quality flags (should be integer).
    - bit_nums: list of bit numbers to create a mask.

    Returns:
    - mask_array: combined mask as a boolean array.
    """
    # Ensure quality_data is of an integer type to support bitwise operations
    if not np.issubdtype(quality_data.dtype, np.integer):
        quality_data = quality_data.astype(int)

    # Initialize the mask_array assuming quality_data has shape (1, H, W)
    mask_array = np.zeros(quality_data.shape[1:], dtype=bool)  # (H, W)

    for bit in bit_nums:
        # Create a temporary mask for each bit
        mask_temp = (quality_data & (1 << bit)) > 0  # Logical condition to flag the bit

        # Combine the temporary mask with the main mask
        mask_array = np.logical_or(mask_array, mask_temp.squeeze())  # Ensure matching dimensions

    return mask_array


username = '2021206190012@whu.edu.cn'
password = '68867FB666lll@'

for index, station in stations.iterrows():
    station_id = station['Site_Id']
    bbox = tuple(station.geometry.bounds)  # (min_lon, min_lat, max_lon, max_lat)

    print(f"Processing Station ID: {station_id}, Bounding Box: {bbox}")

    for year in years:
        print(f"Processing {station_id} for the year {year}...")

        # 创建每个站点每年数据的文件夹
        station_year_folder = os.path.join(output_dir, station_id, str(year))
        os.makedirs(station_year_folder, exist_ok=True)
        print(f"Folder ready: {station_year_folder}")

        # 下载 HLS 数据
        granules = download_hls_data(bbox, year)  # Download HLS data

        # 获取 URL 链接
        hls_results_urls = [granule.data_links() for granule in granules]
        print(hls_results_urls)

        # 处理每个产品的多个波段
        for product_links in hls_results_urls:
            product_name = extract_band_and_product_name(product_links[0])
            station_year_product_folder = os.path.join(station_year_folder, product_name)
            os.makedirs(station_year_product_folder, exist_ok=True)

            subbands_links = extract_bands(product_links)
            print(subbands_links)

            # Load HLS COGs into Memory
            scale_and_mask(subbands_links)
            print(subbands_links[0])

            fsUTM = proj_transfer(subbands_links[0])
            print(fsUTM)

            # Process the subbands using the current station and year information
            process_subbands(subbands_links, fsUTM, station, username, password, station_year_product_folder)
        # print(scale_bands)
        # # 根据scale_bands里名为Fmask波段设置mask
        # bit_nums = [1,2,3,4,5]
        # mask_layer = create_quality_mask(scale_bands[-1].data, bit_nums)
        #   # Example, assuming the first band is your mask
        # for i in range(0, len(scale_bands)-1):
        #     data_to_mask = scale_bands[i]
        #     # Apply masking logic here, using mask_band to mask data_to_mask
        #     masked_data = data_to_mask.where(mask_layer == 1)  # Example using a condition
        # print(masked_data.shape)
        # export_masked_data_as_tif(masked_data, output_filename)



# # # Read the station information from the CSV or the provided data
# # stations = pd.DataFrame({
# #     'Id': ['CA-ARB', 'CA-ARF', 'CA-Ca1', 'CA-Ca2', 'CA-Ca3', 'CA-Cbo', 'CA-Cha', 'CA-DB2', 'CA-DBB', 'CA-ER1', 'CA-LP1',
# #            'CA-MA1', 'CA-MA2'],
# #     'Latitude_d': [52.695, 52.7008, 49.8673, 49.8705, 49.5346, 44.3167, 45.8847, 49.119, 49.1293, 43.6405, 55.1119,
# #                    50.1645, 50.171],
# #     'Longitude_': [-83.9452, -83.955, -125.3336, -125.2909, -124.9004, -79.9333, -67.3569, -122.9951, -122.9849,
# #                    -80.4123, -122.8414, -97.8762, -97.8762]
# # })
# #
# # # Create directory for each station and year
# # output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/tif/'
# #
# # if not os.path.exists(output_dir):
# #     os.makedirs(output_dir)
# #
# # # Define time range
# # years = range(2018, 2019)
# #
# # field = gp.read_file("E:/The North America ecosystem carbon flux/Region_data/geoshp/gadm41_USA_1-20240408T020826Z-001/gadm41_USA_1/gadm41_USA_1_Iowa.json")
# # bbox = tuple(list(field.total_bounds))
# # temporal = ("2018-01-01T00:00:00", "2019-12-31T23:59:59")
# # results = earthaccess.search_data(
# #     short_name=['HLSL30','HLSS30'],
# #     # bounding_box=bbox,
# #     bounding_box=(-83.9452 - 1, 52.695 - 1, -83.9452 + 1, 52.695 + 1),
# #     temporal=temporal,
# #     count=100
# # )
# #
# # hls_results_urls = [granule.data_links() for granule in results]
# # print(hls_results_urls)
# #
# #
# #
# import os
# import earthaccess
# import rioxarray as rxr
# import numpy as np
# import pandas as pd
# import requests
# import rioxarray
# from rasterio.transform import from_origin
# from rasterio.windows import Window
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
#
# # Authenticate using Earthaccess
# auth = earthaccess.login(strategy="netrc")
#
# import os
# import pandas as pd
# import requests
# import time
# import rioxarray
#
# # 从 CSV 文件读取站点数据
# stations_file = "E:/HLS carbon flux/data/sites_263_shp/Carbon flux sites_263.csv"  # 替换为实际的 CSV 文件路径
# stations = pd.read_csv(stations_file)
#
# # 定义输出目录
# output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/tif'
# os.makedirs(output_dir, exist_ok=True)
#
# # 定义年份范围
# years = range(2017, 2019)
#
#
# # 下载300m * 300m 和处理 HLS 数据
# def download_hls_data(lat, lon, year):
#     # 30米分辨率，3个像素对应300米，边界框需要调整为使站点在中心
#     lat_delta = 0.0027  # 300米对等的纬度变化
#     lon_delta = 0.0031  # 300米对等的经度变化
#
#     bounding_box = (
#         lon - lon_delta / 2,  # 左边界
#         lat - lat_delta / 2,  # 下边界
#         lon + lon_delta / 2,  # 右边界
#         lat + lat_delta / 2  # 上边界
#     )
#
#     granules = earthaccess.search_data(
#         short_name=['HLSL30', 'HLSS30'],
#         bounding_box=bounding_box,  # 中心像素正好在站点
#         temporal=(f"{year}-01-01T00:00:00", f"{year}-12-31T23:59:59"),
#     )
#     return granules
#
#
# # 提取波段名称和产品名
# def extract_band_and_product_name(url):
#     base_name = os.path.basename(url)
#     parts = base_name.split('.')
#     product_name = ''.join(parts[1:-4])  # 提取产品名
#     return product_name
#
#
# def download_file_with_retry(url, retries=3, delay=5):
#     for attempt in range(retries):
#         try:
#             response = requests.get(url)
#             response.raise_for_status()  # 如果请求返回码不是200，将引发HTTPError
#             return response.content
#         except requests.RequestException as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             time.sleep(delay)  # 等待一段时间后重试
#     return None  # 如果所有重试都失败
#
# # 遍历站点和年份
# for index, station in stations.iterrows():
#     station_id = station['Id']
#     lat = station['Latitude_d']
#     lon = station['Longitude_']
#
#     for year in years:
#         print(f"Processing {station_id} for the year {year}...")
#
#         station_year_folder = os.path.join(output_dir, station_id, str(year))
#         os.makedirs(station_year_folder, exist_ok=True)
#         print(f"Folder ready: {station_year_folder}")
#
#         # 下载 HLS 数据
#         granules = download_hls_data(lat, lon, year)
#
#         # 获取 URL 链接
#         hls_results_urls = [granule.data_links() for granule in granules]
#
#         # 处理每个产品下载
#         for product_links in hls_results_urls:
#             product_name = extract_band_and_product_name(product_links[0])
#             station_year_product_folder = os.path.join(station_year_folder, product_name)
#             os.makedirs(station_year_product_folder, exist_ok=True)
#             print(f"Folder ready for product: {station_year_product_folder}")
#
#             # 下载每个 TIFF 文件
#             for url in product_links:
#                 content = download_file_with_retry(url)
#                 if content is not None:
#                     file_name = url.split('/')[-1]
#                     file_path = os.path.join(station_year_product_folder, file_name)
#
#                     with open(file_path, 'wb') as f:
#                         f.write(content)
#                     print(f'Successfully downloaded: {file_name}')
#                 else:
#                     print(f'Failed to download {url} after retries.')
#
#             # da_list = [rioxarray.open_rasterio(url) for url in product_links]
#             # combined_da = xr.concat(da_list, dim='band')
#
#
#             # from concurrent.futures import ThreadPoolExecutor
#             # with ThreadPoolExecutor() as executor:
#             #     da_list = list(executor.map(load_raster, product_links))
#             # combined_da = xr.concat(da_list, dim='band')
# #
# #             # 以窗口大小提取数据
# #             subdata = window_subdata_batch(combined_da)
# #             data = subdata.values  # 直接获取 NumPy 数组
# #
# #             # 当前窗口的尺寸应为 (波段数, 5, 5)
# #             print(subdata.shape)
# #
# #             # 获取原始的空间变换信息
# #             transform = combined_da.rio.transform()
# #
# #             # 计算新的变换信息
# #             new_transform = from_origin(
# #                 transform[2] + (subdata.sizes['x'] // 2) * abs(transform[0]),  # X坐标
# #                 transform[3] - (subdata.sizes['y'] // 2) * abs(transform[4]),  # Y坐标
# #                 abs(transform[0]),  # 像素宽度
# #                 abs(transform[4])  # 像素高度
# #             )
# #
# #             # 获取保存的文件名
# #             output_file = os.path.join(station_year_folder,product_name+'.tif')  # 假设基于第一个链接提取名称
# #
# #             # 保存窗口数据为 GeoTIFF
# #             with rio.open(
# #                     output_file,
# #                     'w',
# #                     driver='GTiff',
# #                     height=subdata.shape[1],
# #                     width=subdata.shape[2],
# #                     count=subdata.shape[0],
# #                     dtype=subdata.dtype,  # 从 subdata 获取 dtype
# #                     crs=combined_da.rio.crs,  # 保留原始坐标参考系统
# #                     transform=new_transform
# #             ) as dst:
# #                 dst.write(data)  # 写入 data 而非 subdata
# #
# #             print(f"窗口数据已成功保存为 {output_file}")
# #
# # #         # 将相同产品的文件分组
# # #         product_files = {}
# # #         for granule in granules:
# # #             if 'Collection' in granule:
# # #                 urls = granule['Collection'].get('Data', [])
# # #                 for url in urls:
# # #                     band_name, product_name = extract_band_and_product_name(url)
# # #                     if product_name not in product_files:
# # #                         product_files[product_name] = {}
# # #                     if band_name not in product_files[product_name]:
# # #                         product_files[product_name][band_name] = []
# # #                     product_files[product_name][band_name].append(url)
# # #
# # #         # 处理每个产品的文件
# # #         for product_name, bands in product_files.items():
# # #             # 创建文件夹结构
# # #             station_year_folder = os.path.join(output_dir, station_id, str(year), product_name)
# # #             os.makedirs(station_year_folder, exist_ok=True)
# # #
# # #             for band_name, urls in bands.items():
# # #                 for url in urls:
# # #                     # 下载数据并提取10x10窗口。
# # #                     response = requests.get(url, allow_redirects=True)
# # #                     if response.status_code == 200:
# # #                         file_name = os.path.basename(url)
# # #                         # 保存文件
# # #                         file_path = os.path.join(station_year_folder, file_name)
# # #                         with open(file_path, 'wb') as file:
# # #                             file.write(response.content)
# # #                         print(f"Saved: {file_path}")
# # #                     else:
# # #                         print(f"Failed to download {url}: {response.status_code}")
# # #
# # # print("所有站点处理完成。")
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # #
# # # # 读取站点信息
# # # stations = pd.DataFrame({
# # #     'Id': ['CA-ARB', 'CA-ARF', 'CA-Ca1', 'CA-Ca2', 'CA-Ca3', 'CA-Cbo', 'CA-Cha', 'CA-DB2', 'CA-DBB',
# # #            'CA-ER1', 'CA-LP1', 'CA-MA1', 'CA-MA2'],
# # #     'Latitude_d': [52.695, 52.7008, 49.8673, 49.8705, 49.5346, 44.3167, 45.8847, 49.119, 49.1293,
# # #                    43.6405, 55.1119, 50.1645, 50.171],
# # #     'Longitude_': [-83.9452, -83.955, -125.3336, -125.2909, -124.9004, -79.9333, -67.3569,
# # #                    -122.9951, -122.9849, -80.4123, -122.8414, -97.8762, -97.8762]
# # # })
# # #
# # # # 设置输出目录
# # # output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/tif/'
# # # os.makedirs(output_dir, exist_ok=True)
# # #
# # # # 定义时间范围
# # # years = range(2017, 2019)
# # #
# # #
# # # # 创建文件夹结构
# # # def create_folder_structure(station_id, year, product):
# # #     station_dir = os.path.join(output_dir, station_id, str(year), product)
# # #     os.makedirs(station_dir, exist_ok=True)
# # #     return station_dir
# # #
# # #
# # # # 下载和处理 HLS 数据
# # # def download_hls_data(station, year):
# # #     lat, lon = station['Latitude_d'], station['Longitude_']
# # #
# # #     # 搜索 HLS 数据
# # #     granules = earthaccess.search_data(
# # #         short_name=['HLSL30', 'HLSS30'],
# # #         bounding_box=(lon - 1, lat - 1, lon + 1, lat + 1),
# # #         temporal=(f"{year}-01-01T00:00:00", f"{year+1}-12-31T23:59:59"),
# # #     )
# # #
# # #     if not granules:
# # #         print(f"No data found for station {station['Id']} in {year}. Skipping...")
# # #         return []
# # #
# # #     return granules
# # #
# # #
# # # # 提取5x5像素窗口
# # # def extract_5x5_window(raster_file, station_lat, station_lon):
# # #     with rxr.open_rasterio(raster_file, masked=True) as data:
# # #         transform = data.rio.transform()
# # #         row, col = ~transform * (station_lon, station_lat)
# # #         row, col = int(row), int(col)
# # #         window = Window(col_off=col - 2, row_off=row - 2, width=5, height=5)  # 5x5窗口
# # #         data_window = data.read(window=window)
# # #
# # #     return data_window
# # #
# # #
# # # # 提取波段名称和产品名
# # # def extract_band_and_product_name(url):
# # #     base_name = os.path.basename(url)  # 获取文件名部分
# # #     parts = base_name.split('.')
# # #     band_name = parts[-2]  # 最后一个“.”之前的部分
# # #     product_name = '.'.join(parts[:-3])  # 除去最后两部分，构建产品名
# # #     return band_name, product_name
# # #
# # #
# # # # 遍历站点和年份
# # # for index, station in stations.iterrows():
# # #     station_id = station['Id']
# # #
# # #     for year in years:
# # #         print(f"Processing {station_id} for the year {year}...")
# # #
# # #         # 下载 HLS 数据
# # #         granules = download_hls_data(station, year)
# # #
# # #         if not granules:
# # #             continue
# # #
# # #         # 将相同产品的文件分组
# # #         product_files = {}
# # #         for granule in granules:
# # #             urls = granule['Collection']['Data']
# # #             for url in urls:
# # #                 band_name, product_name = extract_band_and_product_name(url)
# # #                 if product_name not in product_files:
# # #                     product_files[product_name] = {}
# # #                 if band_name not in product_files[product_name]:
# # #                     product_files[product_name][band_name] = []
# # #                 product_files[product_name][band_name].append(url)
# # #
# # #         # 处理每个产品的文件
# # #         for product_name, bands in product_files.items():
# # #             # 创建文件夹结构
# # #             folder = create_folder_structure(station_id, year, product_name)
# # #
# # #             for band_name, urls in bands.items():
# # #                 for url in urls:
# # #                     # 提取5x5窗口
# # #                     data_window = extract_5x5_window(url, station['Latitude_d'], station['Longitude_'])
# # #
# # #                     # 保存数据为 GeoTIFF 文件
# # #                     output_file = os.path.join(folder, f"{band_name}_5x5.tif")
# # #                     data_window.rio.to_raster(output_file)
# # #
# # #                     print(f"Saved: {output_file}")
# # #
# # # print("所有站点处理完成。")