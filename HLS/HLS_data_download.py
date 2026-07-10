
import os
import requests
import rioxarray
import time
import geopandas as gp
import earthaccess
import gc
import geopandas as gpd
import rasterio
import rioxarray as rxr

# Authenticate using Earthaccess
earthaccess.login(strategy="netrc")
# earthaccess.login(persist=True)

stations_file = "E:/HLS carbon flux/data/Sites data/Ameriflux_site_csv_shp/Ameriflux_site_3.5km.geojson"  # 替换为实际的 CSV 文件路径
# stations_file = "E:/HLS carbon flux/data/Sites data/sites_263_shp/sites_263_3km_buffer.geojson"
stations = gp.read_file(stations_file)

output_dir = 'E:/HLS carbon flux/data/Sites data/HLS_Time_series/images/'

os.makedirs(output_dir, exist_ok=True)

# 定义年份范围
years = range(2013, 2025)

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
        # serface_bands = ['B01', 'B02', 'B03', 'B04','B8A', 'B11', 'B12', 'Fmask'] # Coastal/Aerosol Blue Green Red NIR Narrow SWIR1 SWIR2 for L30
        serface_bands = ['B01','B02', 'B03','B04','B05','B06','B07','B09','B10','B11','Fmask','SZA','SAA','VZA','VAA']  # Coastal/Aerosol Blue Green Red NIR Narrow SWIR1 SWIR2 for L30
    else:
        # serface_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'Fmask'] # Coastal/Aerosol Blue Green Red NIR Narrow SWIR1 SWIR2 for L30
        serface_bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12','Fmask','SZA','SAA','VZA','VAA']
     # Subset the assets in the item down to only the desired bands
    for a in product_links:
        if any(b in a for b in serface_bands):
            serface_band_links.append(a)
    return serface_band_links

# Define function to scale
def scaling(band):
    scale_factor = band.attrs['scale_factor']
    band_out = band.copy()
    band_out.data = band.data*scale_factor
    band_out.attrs['scale_factor'] = 1
    return(band_out)

# def get_utm_zone(lon, lat):
#     """根据经纬度获取对应的UTM投影区域代码（EPSG）。"""
#     zone_number = int((lon + 180) / 6) + 1
#     if lat >= 0:
#         return f'EPSG:326{zone_number:02d}'  # 北半球
#     else:
#         return f'EPSG:327{zone_number:02d}'  # 南半球
#
# def clip_by_station(file_path, station):
#     # 打开栅格数据
#     with rasterio.open(file_path) as src:
#         raster_crs = src.crs
#
#     # 从站点几何体中提取经纬度
#     if station.geometry.geom_type == 'MultiPolygon':
#         point = station.geometry.centroid  # 获取 MultiPolygon 的质心
#     else:
#         point = station.geometry
#
#     lon, lat = point.x, point.y
#     utm_epsg = get_utm_zone(lon, lat)  # 获取对应的UTM EPSG
#     print(f'Station {station["Name"]} UTM EPSG: {utm_epsg}')
#
#     # 构建站点的GeoDataFrame并转换CRS
#     first_station_gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326')
#     first_station_transformed = first_station_gdf.to_crs(utm_epsg)
#
#     print("转换后的站点边界:", first_station_transformed.geometry.bounds)
#
#     # 加载栅格数据
#     subband_data = rxr.open_rasterio(file_path)
#     print("栅格数据的边界:", subband_data.rio.bounds())
#     print("数据有效性检查:", subband_data.notnull().sum())
#
#     try:
#         # 执行裁剪
#         cropped_data = subband_data.rio.clip(first_station_transformed.geometry, all_touched=True)
#         return cropped_data
#     except rioxarray.exceptions.NoDataInBounds as e:
#         print(f"裁剪失败: {str(e)}，请确认裁剪范围与数据重叠。")
#         return None

#
#
def clip_by_station(file_path, station):
    try:
        # 确保能正确打开文件
        with rasterio.open(file_path) as src:
            raster_crs = src.crs

        # 检查站点几何有效性
        if station.geometry.is_empty:
            print(f"站点几何为空，无法裁剪: {station}")
            return None

        # 转换坐标系
        first_station_gdf = gpd.GeoDataFrame(geometry=[station.geometry], crs=stations.crs)
        first_station_transformed = first_station_gdf.to_crs(raster_crs)

        # 读取栅格数据
        subband_data = rxr.open_rasterio(file_path)
        if subband_data is None or subband_data.size == 0:
            print(f"读取栅格数据失败: {file_path}")
            return None

        # 执行裁剪
        cropped_data = subband_data.rio.clip(
            first_station_transformed.geometry,
            all_touched=True,
            drop=True  # 自动去除无效区域
        )

        # 检查裁剪结果
        if cropped_data.isnull().all() or cropped_data.size == 0:
            print(f"裁剪后数据为空: {file_path}")
            return None

        return cropped_data

    except Exception as e:  # 捕获所有未处理的异常
        print(f"裁剪过程中发生未预期错误: {str(e)}")
        return None

# Iterate through the stations starting from index 200
for index in range(0, 5):
    print(index)
    station = stations.iloc[index]  # Get the station at the current index
    station_id = station['Name']
    # station_id = station['Site_Id']
    bbox = tuple(station.geometry.bounds)  # (min_lon, min_lat, max_lon, max_lat)
    print("裁剪站点的边界:", station.geometry.bounds)

    years = range(station['AmeriFlux BASE Start'], station['AmeriFlux BASE End'])
    # years = range(2013, 2025)
    for year in years:
        # 根据年份判断
        if year > 2012 and year < 2025:
            site_image_year_name = str(year)
            print(f"Processing {station_id} for the year {year}...")
            # Create directory for each station's yearly data
            station_year_folder = os.path.join(output_dir, station_id, str(year))
            # subset
            os.makedirs(station_year_folder, exist_ok=True)
            # print(f"Folder ready: {station_year_folder}")

            # Download HLS data
            granules = download_hls_data(bbox, year)

            # Get URL links
            hls_results_urls = [granule.data_links() for granule in granules]

            # Process each product's multiple bands
            for product_links in hls_results_urls:
                # print(product_links)
                # subbands_links = extract_bands(product_links)
                subbands_links = product_links

                # print(subbands_links)
                product_name = extract_band_and_product_name(subbands_links[0])
                station_year_product_folder = os.path.join(station_year_folder, product_name)
                os.makedirs(station_year_product_folder, exist_ok=True)

                # print(f"Folder ready for product: {station_year_product_folder}")

                # Main processing loop
                for url in subbands_links:
                    # Define the output file path for clipped content
                    file_name = url.split('/')[-1]
                    clipped_file_path = os.path.join(station_year_product_folder, f"clipped_{file_name}")
                    print(clipped_file_path)
                    # Check if the clipped file already exists
                    if os.path.exists(clipped_file_path):
                        print(f'Skipping download and clipping, file already exists: {clipped_file_path}')
                        continue

                    content = download_file_with_retry(url)

                    if content is not None:
                        file_path = os.path.join(station_year_product_folder, file_name)

                        # Save the downloaded file before clipping
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        # print(f'Successfully downloaded: {file_name}')

                        # Clip content based on the station
                        clip_content = clip_by_station(file_path, station)
                        # print("裁剪后的数据形状:", clip_content.shape)
                        # Save clipped content to a new file
                        # 判断裁剪结果是否有效
                        if clip_content is None:
                            print(f"裁剪失败: {file_path}，跳过")
                            continue

                        # 检查裁剪结果是否包含有效的地理信息
                        if not hasattr(clip_content.rio, "crs") or clip_content.rio.crs is None:
                            print(f"裁剪结果缺少坐标系统信息: {file_path}")
                            continue

                        # 保存到文件
                        try:
                            with rasterio.open(
                                    clipped_file_path,
                                    "w",
                                    driver="GTiff",
                                    height=clip_content.shape[1],  # 确保维度顺序正确
                                    width=clip_content.shape[2],
                                    count=1,
                                    dtype=clip_content.dtype,
                                    crs=clip_content.rio.crs,
                                    transform=clip_content.rio.transform()
                            ) as dst:
                                dst.write(clip_content.squeeze().values, 1)
                            print(f"成功保存: {clipped_file_path}")

                        except Exception as e:
                            print(f"保存文件失败: {str(e)}")
                        # with rasterio.open(clipped_file_path, 'w',
                        #                    driver='GTiff',
                        #                    height=clip_content.shape[1],
                        #                    width=clip_content.shape[2],
                        #                    count=1,  # Adjust for multiple bands if necessary
                        #                    dtype=clip_content.dtype,
                        #                    crs=clip_content.rio.crs,
                        #                    transform=clip_content.rio.transform()) as dst:
                        #     dst.write(clip_content.squeeze().values, 1)

                        # print(f'Successfully clipped and saved: {clipped_file_path}')
                        gc.collect()
                        # 使用shutil.rmtree来删除blocks_path以及其中的所有文件和子目录
                        # os.unlink(file_path)
                        # Delete the original file after clipping
                        try:
                            os.remove(file_path)
                            print(f'Successfully deleted original file: {file_path}')
                        except OSError as e:
                            print(f"Error deleting file {file_path}: {e}")
                    else:
                        print(f'Failed to download {url} after retries.')