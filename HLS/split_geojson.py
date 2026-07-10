import time
import geopandas as gp
from osgeo import gdal
from skimage import io
import geopandas as gpd
import os

stations_file = "E:/HLS carbon flux/data/sites_263_shp/sites_263_3km_buffer.geojson"  # 替换为实际的 CSV 文件路径
stations = gp.read_file(stations_file)

outdir = 'E:/HLS carbon flux/data/sites_263_shp/sites_geojson/'
# Read the GeoJSON file
stations = gpd.read_file(stations_file)

# Ensure the output directory exists
os.makedirs(outdir, exist_ok=True)

# Group by 'Name' and export each group to a separate GeoJSON file
for name, group in stations.groupby('Site_Id'):
    output_file = os.path.join(outdir, f"{name}.geojson")
    group.to_file(output_file, driver='GeoJSON')
    print(f"Exported {name} to {output_file}")