import os
import rasterio
import geopandas as gpd
from rasterio.mask import mask

# 设置路径
tif_folder = "E:/The North America ecosystem carbon flux/Region_data/State_mosaci_geotifs/0216/"  # 输入 TIF 文件夹
input_folder = os.path.join(tif_folder, "State_mosaci_geotifs")  # 存放转换后的 TIF 文件
shp_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/Exclude_lc_without_data/LC_exclude_11_13_16_17.shp"  # 输入 SHP 文件路径
output_folder = os.path.join(tif_folder, "Exclude_by_LC")  # 输出文件夹
# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)
# os.makedirs(output_folder, exist_ok=True)
# 读取裁剪矢量文件
shapefile = gpd.read_file(shp_path)
geoms = [feature["geometry"] for feature in shapefile.to_dict("records")]  # 获取几何边界

# 获取所有 .tif 文件
tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

# 遍历裁剪所有 TIF
for tif_file in tif_files:
    input_tif = os.path.join(input_folder, tif_file)
    output_tif = os.path.join(output_folder, f"Exclude_by_LC_{tif_file}")

    try:
        # 读取 TIF 影像
        with rasterio.open(input_tif) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()

            # 更新元数据
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # 保存裁剪后的 TIF
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"裁剪完成: {tif_file} -> {output_tif}")

    except Exception as e:
        print(f"裁剪失败: {tif_file}, 错误信息: {e}")
