
# Initialize earth engine
import ee
import rasterio as rasterio
from gee_subset import gee_subset
import utility
import geemap
import os
from datetime import datetime, timedelta

service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/GEE_json/biomass-estimates-d58a1f0e77e5.json")
ee.Initialize(credentials)



if __name__ == '__main__':

    # Define the UMC feature collection
    UMC = geemap.geojson_to_ee("E:/Carbon_flux/data/region_shp/gadm41_MEX_1.json/gadm41_MEX_1.json")

    # Get the list of states (NAME_1) within the UMC feature collection
    states_list = UMC.aggregate_array("NAME_1").getInfo()



    # Define the base output directory
    output_base_dir = "E:/Carbon_flux/data/regoin_image/gadm41_MEX_1"

    # Define the list of products and their corresponding bands
    product_list = ['ECMWF/ERA5_LAND/DAILY_AGGR', 'MODIS/006/MCD43A4', 'MODIS/006/MCD15A3H']
    band_list = [
        ['surface_solar_radiation_downwards_sum', 'temperature_2m', 'volumetric_soil_water_layer_1',
         'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'],
        ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2',
         'Nadir_Reflectance_Band3', 'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6',
         'Nadir_Reflectance_Band7','BRDF_Albedo_Band_Mandatory_Quality_Band1', 'BRDF_Albedo_Band_Mandatory_Quality_Band2',
         'BRDF_Albedo_Band_Mandatory_Quality_Band3', 'BRDF_Albedo_Band_Mandatory_Quality_Band4',
         'BRDF_Albedo_Band_Mandatory_Quality_Band5', 'BRDF_Albedo_Band_Mandatory_Quality_Band6',
         'BRDF_Albedo_Band_Mandatory_Quality_Band7'],
        ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC']
    ]

    # Iterate over each state and export products
    for state in states_list[6:]:
        # Create a subdirectory for the current state if it doesn't exist
        state_dir = os.path.join(output_base_dir, state)
        os.makedirs(state_dir, exist_ok=True)

        # Filter UMC by the current state
        state_boundary = UMC.filter(ee.Filter.eq('NAME_1', state))

        # Convert ee.FeatureCollection to ee.Geometry format for the state's boundary
        roi = state_boundary.geometry().bounds()

        # Iterate over each product
        for i, product in enumerate(product_list):
            # Create a subdirectory for the current product if it doesn't exist
            product_name = product.split("/")[-1]
            product_dir = os.path.join(state_dir, product.split("/")[-1])
            os.makedirs(product_dir, exist_ok=True)

            # Define date range
            start_date_str = '2017-01-01'
            end_date_str = '2017-12-31'

            # Convert start and end date strings to datetime objects
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Iterate over the date range for the current product
            current_date = start_date
            while current_date <= end_date:
                current_date_str = current_date.strftime('%Y-%m-%d')

                collection = (ee.ImageCollection(product)
                              .filterDate(current_date_str, (current_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                              .filterBounds(UMC.filter(ee.Filter.eq('NAME_1', state)))
                              .select(band_list[i]))

                # image = collection.first()
                image = collection.mosaic()

                if image is not None:
                    # Generate the image
                    composite = image.clip(roi).unmask()

                    # Export the image to the product-specific subdirectory
                    out_tif = os.path.join(product_dir, f"{current_date_str}.tif")
                    geemap.ee_export_image(composite, filename=out_tif, region=roi, crs='EPSG:4326', scale=500,
                                           file_per_band=False)

                # Move to the next date
                current_date += timedelta(days=1)

