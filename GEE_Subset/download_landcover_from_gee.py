import ee
import rasterio as rasterio

service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/GEE_json/biomass-estimates-d58a1f0e77e5.json")
ee.Initialize(credentials)

# 定义感兴趣区域
roi = ee.FeatureCollection("projects/ascendant-baton-367709/assets/North_America_boundaries")  # 替换成你的用户名

# 定义影像收藏
modisCollection = ee.ImageCollection('MODIS/006/MCD12Q1')

# 筛选土地覆盖数据
landcover = modisCollection.filterDate('2017-01-01', '2017-12-31').mosaic()  # 选择指定日期范围内的第一幅影像

# 裁剪到感兴趣区域
landcover_roi = landcover.clip(roi)

# 下载土地覆盖数据
task = ee.batch.Export.image.toDrive(image=landcover_roi,
                                     description='landcover_export',
                                     folder='E:/Carbon_flux/data/MODIS Land Cover',
                                     region= roi.geometry().bounds(),
                                     scale=500,
                                     fileFormat='GeoTIFF')
task.start()
