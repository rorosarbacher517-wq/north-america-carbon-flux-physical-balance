import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# 定义文件夹路径
# geojson_folder = 'E:\The North America ecosystem carbon flux\Region_data\geoshp\gadm41_CAN_1-20240306T074436Z-001\gadm41_CAN_1'
geojson_folder = 'E:/The North America ecosystem carbon flux/Region_data/geoshp/gadm41_USA_1-20240408T020826Z-001/gadm41_USA_1/'
# geojson_folder ='E:/The North America ecosystem carbon flux/Region_data/geoshp/gadm41_MEX_1-20240306T090223Z-001/gadm41_MEX_1'
raster_folder = 'E:/The North America ecosystem carbon flux/Region_data/block_geotifs_merge_to_state_1013/block_geotifs_merge_to_state/'
output_folder = 'E:/The North America ecosystem carbon flux/Region_data/State_mosaci_geotifs/0216/State_clip/'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for geojson_file in os.listdir(geojson_folder):
    # if geojson_file.endswith('.json'):  # 根据新的扩展名检查.geojson
    if geojson_file.endswith('.geojson'):  # 根据新的扩展名检查.geojson
        # 提取州名，获取最后一个'_'之后的部分
        state_name = geojson_file.split('_')[-1].split('.')[0]
        geojson_path = os.path.join(geojson_folder, geojson_file)  # 完整路径

        # 读取对应的 GeoJSON
        gdf = gpd.read_file(geojson_path)

        # 找到对应的栅格文件夹
        raster_path = os.path.join(raster_folder, state_name)

        if os.path.exists(raster_path):
            for raster_file in os.listdir(raster_path):
                if raster_file.endswith('.tif'):
                    raster_file_path = os.path.join(raster_path, raster_file)

                    # 裁剪栅格文件
                    with rasterio.open(raster_file_path) as src:
                        out_image, out_transform = mask(src, gdf.geometry, crop=True)
                        out_meta = src.meta.copy()  # 更新元数据

                        # 更新元数据
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })

                        # 去掉.tif后缀名，构建输出文件名
                        raster_file_name = os.path.splitext(raster_file)[0]  # 去除后缀
                        output_file_name = f"{state_name}_{raster_file_name}_clip.tif"
                        output_file_path = os.path.join(output_folder, output_file_name)

                        # 写入裁剪后的栅格文件
                        with rasterio.open(output_file_path, "w", **out_meta) as dest:
                            dest.write(out_image)
                        print(f"裁剪完成: {output_file_path}")
        else:
            print(f"没有找到州 {state_name} 的栅格文件夹，跳过该州。")