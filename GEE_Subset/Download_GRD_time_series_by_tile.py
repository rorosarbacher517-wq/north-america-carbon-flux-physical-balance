import ee
import geemap
import os
from datetime import datetime, timedelta
from datetime import timedelta
import rasterio
import numpy as np

service_account = 'gpp-estimation-dl@gpp-estimation.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/个人重要文件/GEE_APIKER/GPP_estimation_DL/gpp-estimation-a13404c05e60.json")
ee.Initialize(credentials)

# Input and output paths
UMC_input_path = 'E:/Sentinel SAR/region_tile/'
output_base_dir = 'E:/Sentinel SAR/SAR_data/tile_images'

# Load the tile metadata from the JSON file
json_file_path = os.path.join(UMC_input_path, 'Selected_tiles.geojson')
UMC = geemap.geojson_to_ee(json_file_path)
tile_names = UMC.aggregate_array("Name").getInfo()

# Define the list of products and their corresponding bands
# Define band list and date range
band_list = ['VV', 'VH', 'HH', 'HV', 'angle']
start_date_str = '2022-01-01'
end_date_str = '2022-12-31'

# Convert start and end date strings to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')


## Iterate over each tile
for tile_name in tile_names:
    # Filter to get the specific tile geometry using name
    tile_geometry = UMC.filter(ee.Filter.eq("Name", tile_name)).geometry()

    # Iterate over the date range
    current_date = start_date
    while current_date <= end_date:
        # Convert the current date to a string in the format 'YYYY-MM-dd'
        current_date_str = current_date.strftime('%Y-%m-%d')

        # Create a subdirectory for the current tile and year
        year_dir = os.path.join(output_base_dir, tile_name, str(current_date.year))
        os.makedirs(year_dir, exist_ok=True)

        for band in band_list:
            # Define the output file path for the .tif file
            out_tif = os.path.join(year_dir, f"{current_date_str}_{band}.tif")

            # Check if the .tif file already exists
            if os.path.exists(out_tif):
                print(f"{out_tif} already exists. Skipping download for {tile_name} on {current_date_str} for band {band}.")
                continue  # Skip to the next band if the file exists

            # Create the image collection, filtering by date and bounds
            collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                          .filterDate(current_date_str, (current_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                          .filterBounds(tile_geometry))

            # Check if the collection has any images
            image_count = collection.size().getInfo()
            if image_count == 0:
                print(f"No images found for {tile_name} on {current_date_str} for band {band}.")
                continue  # Skip to the next band if no images found

            # Check if the band exists in the first image
            first_image = collection.first()
            available_bands = first_image.bandNames().getInfo()
            if band not in available_bands:
                print(f"Band '{band}' not available for {tile_name} on {current_date_str}. Available bands: {available_bands}.")
                continue  # Skip to the next band if band not available

            # Get the first image in the collection for the current band
            image = collection.select(band).mosaic().clip(tile_geometry).unmask()

            # Convert image to float32
            image = image.toFloat()

            # Export the image to the specified subdirectory
            geemap.download_ee_image(
                image=image,
                filename=out_tif,
                region=tile_geometry,
                crs="EPSG:4326",
                scale=10,
                resampling='near'
            )

        # Move to the next date
        current_date += timedelta(days=1)
