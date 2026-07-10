import os
import geopandas as gpd
import rioxarray as rxr
import rasterio
from tqdm import tqdm
import gc


def clip_by_station(file_path, station_geometry, src_crs):
    """优化的裁剪函数"""
    try:
        # 将几何对象包装为GeoDataFrame
        station_gdf = gpd.GeoDataFrame(
            geometry=[station_geometry],
            crs=src_crs  # 原始坐标系
        ).to_crs(rxr.open_rasterio(file_path).rio.crs)  # 目标坐标系

        # 执行裁剪
        with rxr.open_rasterio(file_path, masked=True) as src:
            clipped = src.rio.clip(
                station_gdf.geometry,
                all_touched=True,
                drop=True
            )

            if clipped.isnull().all():
                print(f"⚠️ 裁剪结果为空: {os.path.basename(file_path)}")
                return None
            return clipped
    except Exception as e:
        print(f"裁剪失败 {os.path.basename(file_path)}: {str(e)}")
        return None


# ================ 配置参数 ================
geojson_path = r"E:/HLS carbon flux/data/Sites data/Ameriflux_site_csv_shp/Ameriflux_site_3.5km.geojson"
base_dir = r"E:/HLS carbon flux/data/Sites data/HLS_Time_series/images"
# ========================================

# 加载地理信息（显式设置CRS）
print("正在加载地理信息...")
sites_gdf = gpd.read_file(geojson_path).set_index('Name')
sites_gdf = sites_gdf.set_crs(epsg=4326, allow_override=True)  # 确保原始CRS正确

# 遍历处理指定站点（示例处理索引3的站点）
site_list = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])

for site_name in site_list:  # 处理第4个站点
    site_path = os.path.join(base_dir, site_name)
    print(f"\n=== 正在处理站点: {site_name} ===")

    try:
        # 获取原始几何对象及其CRS
        raw_geometry = sites_gdf.loc[site_name].geometry
        src_crs = sites_gdf.crs  # 获取GeoJSON的原始坐标系
    except KeyError:
        print(f"站点 {site_name} 未在GeoJSON中找到，跳过...")
        continue

    # 遍历所有HLS文件
    for root, _, files in os.walk(site_path):
        for file in files:
            if not (file.startswith("HLS") and file.endswith(".tif")):
                continue

            input_path = os.path.join(root, file)
            output_path = os.path.join(root, f"clipped_{file}")

            # if os.path.exists(output_path):
            #     continue

            # 执行裁剪
            clipped_data = clip_by_station(input_path, raw_geometry, src_crs)
            if clipped_data is None:
                continue

            # 保存结果
            try:
                clipped_data.rio.to_raster(
                    output_path,
                    driver="GTiff",
                    compress="LZW",
                    dtype=clipped_data.dtype
                )
                os.remove(input_path)
                print(f"成功处理: {file}")
            except Exception as e:
                print(f"保存失败 {file}: {str(e)}")

            gc.collect()

print("\n处理完成！")