import os
import geopandas as gpd  # 引入 geopandas
import ee
import geemap  # 确保 geemap 已安装


# ee.Authenticate()
# ee.Initialize(project='ee-grn')
service_account = 'gpp-estimation-dl@gpp-estimation.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/个人重要文件/GEE_APIKER/GPP_estimation_DL/gpp-estimation-a13404c05e60.json")
ee.Initialize(credentials)

shp_path = 'E:/The North America ecosystem carbon flux/Region_data/geoshp/gadm41_USA_1-20240408T020826Z-001/gadm41_USA_1/gadm41_USA_1_Alabama.json'
# 使用 geopandas 加载 GeoJSON 数据
gdf = gpd.read_file(shp_path)
# 将 GeoPandas DataFrame 转换为 Earth Engine FeatureCollection
roi = ee.FeatureCollection(gdf.__geo_interface__)
# roi = ee.FeatureCollection('projects/ee-grn/assets/241208-15TVH')
image = ee.ImageCollection('USDA/NASS/CDL').\
    filter(ee.Filter.date('2019-01-01', '2020-12-31'))\
    .first()

save_dir = r"E:\cropland"
filename = "CDL_15-TVH.tif"
file_path = os.path.join(save_dir, filename)

os.makedirs(save_dir, exist_ok=True)
geemap.ee_export_image(
    image,
    filename=file_path,
    scale=10,
    region=roi.geometry(),
    file_per_band=False,
    crs="EPSG:4326"
)