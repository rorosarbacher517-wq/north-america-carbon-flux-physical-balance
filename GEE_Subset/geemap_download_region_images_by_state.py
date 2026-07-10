# Initialize earth engine
import ee
import geemap
import os
from datetime import datetime, timedelta
from datetime import timedelta

service_account = 'gpp-estimation-dl@gpp-estimation.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "/dat1/Fanbin_data/Vegetation_productivity_prediction/GEE_API_Key/gpp-estimation-a13404c05e60.json")
ee.Initialize(credentials)


if __name__ == '__main__':
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
    # Define the base output directory
    output_base_dir = "/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/original_images/gadm41_USA_1/"
    # Define the UMC feature collection
    UMC_input_path = "/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/North_America_Bounds/gadm41_USA_1/"
    UMC_list = os.listdir(UMC_input_path)
    for j in range(1, 2):
        UMC_path = os.path.join(UMC_input_path, UMC_list[j])
        UMC = geemap.geojson_to_ee(UMC_path)
        state = UMC.aggregate_array("NAME_1").getInfo()
        print(state)
        state_str = state[0]
        # Create a subdirectory for the current state if it doesn't exist
        state_dir = os.path.join(output_base_dir, state_str)
        os.makedirs(state_dir, exist_ok=True)
        roi_bounds = UMC.filter(ee.Filter.eq('NAME_1', state_str)).geometry().bounds()
        # Define date range
        start_date_str = '2017-01-01'
        end_date_str = '2017-01-01'
        # Convert start and end date strings to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        # Iterate over the date range
        current_date = start_date
        while current_date <= end_date:
            # Convert the current date to a string in the format 'YYYY-MM-dd'
            current_date_str = current_date.strftime('%Y-%m-%d')
            for i, product in enumerate(product_list):
                collection = (ee.ImageCollection(product)
                              .filterDate(current_date_str, (current_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                              .filterBounds(roi_bounds)
                              .select(band_list[i]))
                # Check if the product has data for the current date
                size = collection.size().getInfo()
                # If there is data, continue processing and save the output, then move to the next product
                if size > 0:
                    # Get the first image in the collection
                    image = collection.mosaic()
                    # Generate the composite image
                    composite = image.clip(roi_bounds).unmask()
                    # Create a subdirectory for the current product if it doesn't exist
                    product_dir = os.path.join(state_dir, product.split("/")[-1])
                    os.makedirs(product_dir, exist_ok=True)
                    # Export the image to the product-specific subdirectory
                    out_tif = os.path.join(product_dir, f"{current_date_str}.tif")
                    geemap.download_ee_image(
                        image=composite,
                        filename=out_tif,
                        region=roi_bounds,
                        crs="EPSG:4326",
                        scale=500,
                        resampling='near'
                    )
                # If there is no data, skip processing this product
                else:
                    continue
            # Move to the next date
            current_date += timedelta(days=1)