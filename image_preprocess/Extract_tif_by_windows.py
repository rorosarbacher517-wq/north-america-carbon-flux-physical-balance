# -*- coding: utf-8 -*-
import os
import json
import geopandas as gpd
from osgeo import gdal
import rasterio
from rasterio.mask import mask

# 读取JSON文件
input_folder = "E:/The North America ecosystem carbon flux/Region_data/region_shp/gadm41_MEX_1.json/gadm41_MEX_1.json"
output_folder = 'E:/The North America ecosystem carbon flux/Region_data/State_clip/'
raster_folder = 'E:/The North America ecosystem carbon flux/Region_data/block_geotifs_merge_to_state_1013_402/'

# with open(input_folder, 'r') as file:
#     data = json.load(file)
# 以 UTF-8 编码打开并读取 GeoJSON 文件
with open(input_folder, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 读取内容

    # 检查每个州名
    for feature in data['features']:
        state_name = feature['properties']['NAME_1']
        print(state_name)  # 直接打印，检查是否正常
# # Iterate over each state
# for i in range(len(data['features'])):
#     state_name = data['features'][i]['properties']['NAME_1']
#     print(state_name)
#     geo = [data['features'][i]['geometry']]
for feature in data['features']:
    state_name = feature['properties']['NAME_1']

    # Print the state name ensuring UTF-8 handling
    print(state_name.encode('utf-8').decode('utf-8'))  # This may not be necessary, but ensures encoding safety.

    # Store the geometry
    geo = [feature['geometry']]

    raster_path = os.path.join(raster_folder,state_name)
    if os.path.exists(raster_path):
        for raster_file in os.listdir(raster_path):
            rasterdata = rasterio.open(os.path.join(raster_path,raster_file))
            out_image, out_transform = mask(rasterdata, geo, all_touched=True, crop=True, nodata=rasterdata.nodata)
            # Crop the TIFF file based on the boundary
            output_path = f'{output_folder}{state_name}_{raster_file[:-4]}_clip.tif'
            # Save the cropped image to the output path
            with rasterio.open(output_path, 'w', driver='GTiff',
                               width=out_image.shape[2], height=out_image.shape[1],
                               count=out_image.shape[0], dtype=out_image.dtype,
                               crs=rasterdata.crs, transform=out_transform) as dst:
                dst.write(out_image)

                print(f'Saved cropped image to: {output_path}')

