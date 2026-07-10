# Initialize earth engine
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
    output_base_dir = "E:/Carbon_flux/data/regoin_image/gadm41_MEX_1"

    # Define the list of products and their corresponding bands
    # product_list = ['ECMWF/ERA5_LAND/DAILY_AGGR', 'MODIS/006/MCD43A4', 'MODIS/006/MCD15A3H']
    # band_list = [
    #     ['surface_solar_radiation_downwards_sum', 'temperature_2m', 'volumetric_soil_water_layer_1',
    #      'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'],
    #     ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2',
    #      'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6',
    #      'Nadir_Reflectance_Band7', 'BRDF_Albedo_Band_Mandatory_Quality_Band1',
    #      'BRDF_Albedo_Band_Mandatory_Quality_Band2',
    #      'BRDF_Albedo_Band_Mandatory_Quality_Band3', 'BRDF_Albedo_Band_Mandatory_Quality_Band4',
    #      'BRDF_Albedo_Band_Mandatory_Quality_Band5', 'BRDF_Albedo_Band_Mandatory_Quality_Band6',
    #      'BRDF_Albedo_Band_Mandatory_Quality_Band7'],
    #     ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
    # ]
    product_list = ['MODIS/006/MCD15A3H']
    band_list = [
        ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
    ]

    # Iterate over each state and export products
    for state in states_list[6:7]:
        # Create a subdirectory for the current state if it doesn't exist
        state_dir = os.path.join(output_base_dir, state)
        os.makedirs(state_dir, exist_ok=True)

        # Iterate over each product
        for i, product in enumerate(product_list):
            # Create a subdirectory for the current product if it doesn't exist
            product_name = product.split("/")[-1]
            product_dir = os.path.join(state_dir, product.split("/")[-1])
            os.makedirs(product_dir, exist_ok=True)

            # Define date range
            start_date_str = '2017-01-01'
            end_date_str = '2017-03-01'

            # Convert start and end date strings to datetime objects
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Iterate over the date range for the current product
            current_date = start_date
            while current_date <= end_date:
                current_date_str = current_date.strftime('%Y-%m-%d')

                collection = (ee.ImageCollection(product)
                              .filterDate(current_date_str, (current_date + timedelta(days=4)).strftime('%Y-%m-%d'))
                              .filterBounds(UMC.filter(ee.Filter.eq('NAME_1', state)))
                              .select(band_list[i]))

                # image = collection.first()
                image = collection.mosaic()
                composite = image.clip(UMC.filter(ee.Filter.eq('NAME_1', state)).geometry())
                out_tif = os.path.join(product_dir, f"{current_date_str}.tif")

                geemap.download_ee_image(
                    image=composite,
                    filename=out_tif,
                    region=UMC.filter(ee.Filter.eq('NAME_1', state)).geometry().bounds(),
                    crs="EPSG:4326",
                    scale=500,
                )
                current_date += timedelta(days=4)
                if image is not None:
                    composite = image.clip(UMC.filter(ee.Filter.eq('NAME_1', state)).geometry())
                    request_size = composite.projection().nominalScale().multiply(4800).round()  # Estimate request size
                    request_size_num = request_size.getInfo()  # Get the numeric value from the ee.Number object
                    if request_size_num > 50331648:
                        # file_path = get_drive_file_path(state, product)
                        geemap.ee_export_image_to_drive(image=composite,
                                                             description=f"ImageExport_{state}_{product_name}_{current_date_str}",
                                                             folder= 'gadm41_MEX_1', region=UMC.filter(
                                ee.Filter.eq('NAME_1', state)).geometry().bounds(), crs='EPSG:4326', scale=500)
                        # # Get the export link
                        # # # print(task.getDownloadURL())
                        # # # 跳出这个日期循环，直接到下一个产品
                        # print('the total size > 50331648 bit')
                        # break
                    else:
                        out_tif = os.path.join(product_dir, f"{current_date_str}.tif")

                        geemap.ee_export_image(composite, filename=out_tif,
                                               region=UMC.filter(ee.Filter.eq('NAME_1', state)).geometry().bounds(),
                                               crs='EPSG:4326', scale=500, file_per_band=False)

                # Move to the next date
                current_date += timedelta(days=1)

    # # Define the UMC feature collection 文件下载来源https://gadm.org/download_world.html
    # UMC = geemap.geojson_to_ee("E:/Carbon_flux/data/region_shp/gadm41_MEX_1.json/gadm41_MEX_1.json")
    #
    # # Get the list of states (NAME_1) within the UMC feature collection
    # states_list = UMC.aggregate_array("NAME_1").getInfo()
    #
    # # Output base directory
    # output_base_dir = "E:/Carbon_flux/data/regoin_image/era5land_daily"
    #
    # # Iterate over each state and export daily products
    # for state in states_list:  # Restricting to the first 2 states for demonstration purposes
    #     # Create a subdirectory for the current state if it doesn't exist
    #     state_dir = os.path.join(output_base_dir, state)
    #     os.makedirs(state_dir, exist_ok=True)
    #
    #     # Filter UMC by the current state
    #     state_boundary = UMC.filter(ee.Filter.eq('NAME_1', state))
    #
    #     # Convert ee.FeatureCollection to ee.Geometry format for the state's boundary
    #     roi = state_boundary.geometry().bounds()
    #
    #     # Define the image collection
    #     s2 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    #
    #     # Define date range
    #     start_date_str = '2017-01-01'
    #     end_date_str = '2017-12-31'
    #
    #     # Convert start and end date strings to datetime objects
    #     start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    #     end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    #
    #     # Iterate over the date range
    #     current_date = start_date
    #     while current_date <= end_date:
    #         # Convert the current date to a string in the format 'YYYY-MM-dd'
    #         current_date_str = current_date.strftime('%Y-%m-%d')
    #
    #         # Filter and select data for the current date
    #         collection = (
    #             s2.filterDate(current_date_str, (current_date + timedelta(days=1)).strftime('%Y-%m-%d'))
    #             .filterBounds(state_boundary)
    #             .select(['surface_solar_radiation_downwards_sum','temperature_2m','volumetric_soil_water_layer_1',
    #                      'volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4'])
    #         )
    #
    #         # Get the first image in the collection
    #         image = collection.first()
    #
    #         # Generate the image
    #         composite = image.clip(roi).unmask()
    #
    #         # Export the image to the state-specific subdirectory
    #         out_tif = os.path.join(state_dir, f"{current_date_str}.tif")
    #
    #         geemap.ee_export_image(composite, filename=out_tif, region=roi, scale=500, file_per_band=False)
    #
    #         # Move to the next date
    #         current_date += timedelta(days=1)


    # # #将shp转为json
    # UMC = geemap.geojson_to_ee("E:/Carbon_flux/data/region_shp/gadm41_CAN_1.json/gadm41_CAN_1.json")
    #
    # # 将ee.FeatureCollection转换为ee.geometry格式的最小外接矩形
    # roi = UMC.geometry().bounds()
    #
    # s2 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    #
    # collection = s2.filterDate('2017-01-01', '2017-01-03') \
    #     .filterBounds(UMC) \
    #     .select('surface_solar_radiation_downwards_sum',
    #                                             'temperature_2m',
    #                                             'volumetric_soil_water_layer_1',
    #                                             'volumetric_soil_water_layer_2',
    #                                             'volumetric_soil_water_layer_3',
    #                                             'volumetric_soil_water_layer_4')
    # composite = collection.median().clip(roi)
    #
    # work_dir = os.path.join(os.path.expanduser("E:/Carbon_flux/data/regoin_image/era5land_daily"), 'tif')
    # if not os.path.exists(work_dir):
    #     os.makedirs(work_dir)
    #
    # out_tif = os.path.join(work_dir, "S2_SR_2022_huainan.tif")
    #
    # geemap.download_ee_image(
    #     image=composite,
    #     filename=out_tif,
    #     region=roi,
    #     crs="EPSG:4326",
    #     scale=500,
    # )

    # # 上传shp文件到 Earth Engine
    # shp_ee = ee.FeatureCollection('projects/ascendant-baton-367709/assets/North_America_boundaries.shp')
    #
    # # 定义需要下载的数据集
    # dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    #
    # # 定义日期范围
    # start_date = ee.Date('2017-01-01')
    # end_date = ee.Date('2017-12-31')
    #
    # # 筛选指定日期范围内的数据
    # filtered_dataset = dataset.filterDate(start_date, end_date)
    #
    # # 选择需要的波段
    # selected_dataset = filtered_dataset.select(['surface_solar_radiation_downwards_sum',
    #                                             'temperature_2m',
    #                                             'volumetric_soil_water_layer_1',
    #                                             'volumetric_soil_water_layer_2',
    #                                             'volumetric_soil_water_layer_3',
    #                                             'volumetric_soil_water_layer_4'])
    #
    # # 导出数据到云盘
    # task = ee.batch.Export.image.toDrive(image=selected_dataset,
    #                                      description='export_ERA5_data',
    #                                      folder='GEE_exports',
    #                                      region=shp_ee.geometry().bounds(),
    #                                      scale=500)  # 设定分辨率
    #
    # task.start()
    #
    # # 获取结果的下载链接
    # url = selected_dataset.getDownloadURL({
    #     'name': 'export_ERA5_data',
    #     'scale': 500,
    #     'crs': 'EPSG:4326'
    # })