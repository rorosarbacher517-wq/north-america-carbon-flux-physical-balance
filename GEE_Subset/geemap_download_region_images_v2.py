# Initialize earth engine
# 本版本是在初始版本的基础上将获取到的一个州的影像分块输出


import ee
import rasterio as rasterio
from gee_subset import gee_subset
import utility
import geemap
import os
from datetime import datetime, timedelta
from datetime import timedelta

service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/GEE_json/biomass-estimates-d58a1f0e77e5.json")
ee.Initialize(credentials)



if __name__ == '__main__':
    # 定义shp文件在本地的路径
    # shp_path = "E:/Carbon_flux/data/originate_data/North America boundaries/USA_Merge/UMC_Merge.shp"
    #
    # # 读取GeoJSON文件并转换为ee.FeatureCollection
    # UMC = geemap.geojson_to_ee("E:/Carbon_flux/data/originate_data/North America boundaries/USA_Merge/UMC_Merge.geojson")

    # Define the UMC feature collection
    UMC = geemap.geojson_to_ee("E:/Carbon_flux/data/region_shp/gadm41_MEX_1.json/gadm41_MEX_1.json")


    # Get the list of states (NAME_1) within the UMC feature collection
    states_list = UMC.aggregate_array("NAME_1").getInfo()

    # Define the base output directory
    output_base_dir = "E:/Carbon_flux/data/regoin_image/era5land_daily"

    # Define the list of products and their corresponding bands
    product_list = ['ECMWF/ERA5_LAND/DAILY_AGGR', 'MODIS/006/MCD43A4', 'MODIS/006/MCD15A3H']
    band_list = [
        ['surface_solar_radiation_downwards_sum', 'temperature_2m', 'volumetric_soil_water_layer_1',
         'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'],
        ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2',
         'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6',
         'Nadir_Reflectance_Band7', 'BRDF_Albedo_Band_Mandatory_Quality_Band1',
         'BRDF_Albedo_Band_Mandatory_Quality_Band2',
         'BRDF_Albedo_Band_Mandatory_Quality_Band3', 'BRDF_Albedo_Band_Mandatory_Quality_Band4',
         'BRDF_Albedo_Band_Mandatory_Quality_Band5', 'BRDF_Albedo_Band_Mandatory_Quality_Band6',
         'BRDF_Albedo_Band_Mandatory_Quality_Band7'],
        ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
    ]
    # product_list = ['MODIS/006/MCD15A3H']
    # band_list = [
    #     ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
    # ]

    # Iterate over each state and export products
    for state in states_list[6:7]:
        # Create a subdirectory for the current state if it doesn't exist
        state_dir = os.path.join(output_base_dir, state)
        os.makedirs(state_dir, exist_ok=True)

        region = UMC.filter(ee.Filter.eq('NAME_1', state)).geometry()

        # Iterate over each product
        for i, product in enumerate(product_list):
            # Create a subdirectory for the current product if it doesn't exist
            product_name = product.split("/")[-1]
            product_dir = os.path.join(state_dir, product.split("/")[-1])
            os.makedirs(product_dir, exist_ok=True)

            # Define date range
            start_date_str = '2017-01-01'
            end_date_str = '2017-01-02'

            # Convert start and end date strings to datetime objects
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Iterate over the date range for the current product
            current_date = start_date


            while current_date <= end_date:
                current_date_str = current_date.strftime('%Y-%m-%d')

                if i != 2:
                    interval_day = 1
                else:
                    interval_day = 4

                collection = (ee.ImageCollection(product)
                              .filterDate(current_date_str, (current_date + timedelta(days=interval_day)).strftime('%Y-%m-%d'))
                              .filterBounds(UMC.filter(ee.Filter.eq('NAME_1', state)))
                              .select(band_list[i]))

                # image = collection.first()
                image = collection.mosaic()
                composite = image.clip(region)

                # 获取影像行列信息
                # rows = composite.getInfo()
                # cols = composite.getInfo()
                # 获取影像行列信息
                rows = composite.getInfo()['bands'][0]['dimensions'][0]
                cols = composite.getInfo()['bands'][0]['dimensions'][1]

                # rows,cols除以32*32大小的块 确定即将导出的tile的大小
                # 确定即将导出的tile的大小
                tile_size = 32
                if rows < 32:
                    rows_in_tiles = 1
                    tile_size = min(cols, 32)
                else:
                    rows_in_tiles = rows // tile_size

                if cols < 32:
                    cols_in_tiles = 1
                    tile_size = min(rows, 32)
                else:
                    cols_in_tiles = cols // tile_size

                # 生成用于切割影像的网格features
                features = geemap.fishnet(region, rows_in_tiles, cols_in_tiles, delta=0)

                geemap.download_ee_image_tiles(composite,features,product_dir,prefix=current_date_str+'_',crs="EPSG:4326", scale=500,resampling='near')

                current_date += timedelta(days=interval_day)  # 增加日期

                # for tile in tiles:
                #     out_tif = os.path.join(product_dir, f"{current_date_str}_block{block_number}.tif")  # 在原有名称的后面加上块的编号
                #     geemap.download_ee_image(image=tile, filename=out_tif,
                #                              region=UMC.filter(ee.Filter.eq('NAME_1', state)).geometry().bounds(),
                #                              crs="EPSG:4326", scale=500,num_threads=tile_number)
                #     block_number += 1  # 更新块编号
                #
                # current_date += timedelta(days=interval_day)  # 增加日期


