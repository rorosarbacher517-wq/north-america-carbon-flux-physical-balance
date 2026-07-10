# v1.6 Hankui on Jun 29 to plot s1 time series using a shapefile
# v1.5 Hankui on Jun 29 to add shapefile support 
# v1.4 Hankui on Jun 29 2022 to add Sentinel-1 support 
# Hankui on Oct 24 2021 for GEE extract
# https://github.com/bluegreen-labs/gee_subset


## note the key function is col.getRegion it just extract all pixel values in image collection 
## https://developers.google.com/earth-engine/apidocs/ee-imagecollection-getregion
## !!!!!!!!!!! "extract more than 1048576 values will result in an error."
## this is because one cannot simply save ee.ImageCollection to local 
## https://gis.stackexchange.com/questions/363840/download-google-earth-engine-collection-to-local-drive/363846#363846


# import datetime
# start_time = datetime.datetime.now()
# import Pro_gee_subset_v1_3
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)
# 
# import datetime
# start_time = datetime.datetime.now()
# import importlib
# importlib.reload(Pro_gee_subset_v1_3)
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)

import os, re
import pandas as pd
from datetime import datetime
import ee
import gee_subset_master.gee_subset.gee_subset

import importlib
importlib.reload (gee_subset_master.gee_subset.gee_subset)
# import gee_subset-master.gee_subset
# !!!! Hankui on Jun 28, 2022 - I have trouble using ee.Initialize() and trouble to re-Initialize
# refer to https://developers.google.com/earth-engine/guides/python_install-conda
# # earthengine authenticate
# # ee.Authenticate() (DO NOT USE this as it prompt a web browse inside python window not working 
# Initialize earth engine
# ee.Initialize()
## !!!! use below success linked with google cloud account !!!!!
## https://developers.google.com/earth-engine/guides/service_account
import ee
service_account = 'service-account-hankui@sdsu-planet.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '/home/zhangh/common.tools/download/google_cloud/sdsu-planet-f81e092777d9.json')
ee.Initialize(credentials)

## This is a Dam in Nebraska 
lat = 40.897
lon = -96.526
lat = 40.2733
lon = -96.5766
modis_bands = ["Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4"]
Landsat_bands = ["B2", "B3", "B4", "B5"]
Landsat_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"] # revised on Jun 28, 2022 
Sentinel_bands = ["B2", "B3", "B4", "B8A"]
Sentinel1_bands = ['VV', 'VH', 'angle']

# your call (below a MODIS example)
# df = gee_subset.gee_subset.gee_subset(product = "MODIS/MYD09Q1", bands = ["sur_refl_b01", "sur_refl_b02"], start_date = "2015-01-01", end_date = "2015-12-31", latitude = lat, longitude = lon, scale = 30)
# pad is in km 
pad = 0.25
# df_pad_modis = gee_subset.gee_subset.gee_subset(product = "MODIS/MYD09Q1", bands = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04"], start_date = "2015-01-01", end_date = "2015-12-31", pad=pad, latitude = 44, longitude = -72, scale = 30)

pad = 1.5

startdate = "2021-10-01"
enddate   = "2021-12-01"
startdate = "2021-01-01"
enddate   = "2021-12-31"
# product="MODIS/006/MCD43A4"
# bands=modis_bands
# start_date=startdate; end_date=enddate; pad=pad; latitude=lat; longitude=lon; scale=10
# df_pad_modis = gee_subset.gee_subset.gee_subset(product = "MODIS/006/MCD43A4", bands = modis_bands, start_date=startdate, end_date=enddate, pad=pad, latitude = lat, longitude = lon, scale = 10)
product="COPERNICUS/S1_GRD"
bands=Sentinel1_bands
start_date=startdate; end_date=enddate; pad=pad; latitude=lat; longitude=lon; scale=10
instrument = None; orbit=None
# from gee_subset.gee_subset import gee_subset
# import gee_subset.gee_subset
# importlib.reload (gee_subset.gee_subset)
pad = 0.4 ## is the maximun sentinel-1 can tolerate
df_pad_sentinel1_box = gee_subset_master.gee_subset.gee_subset.gee_subset(product="COPERNICUS/S1_GRD"      , bands=Sentinel1_bands, start_date=startdate, end_date=enddate, pad=pad, latitude=lat, longitude=lon, scale=10)
# bands=Landsat_bands
# product="LANDSAT/LC08/C02/T1_L2"
# start_date=startdate; end_date=enddate; pad=pad; latitude=lat; longitude=lon; scale=10
# instrument = None; orbit=None

## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## Too many values: 26027 points x 4 bands x 19 images > 1048576.
# pad = 1.5 ## MAX values for 1.5 
# df_pad_Landsat8 = gee_subset_master.gee_subset.gee_subset.gee_subset(product = "LANDSAT/LC08/C02/T1_L2", bands=Landsat_bands  , start_date=startdate, end_date=enddate, pad=pad, latitude=lat, longitude=lon, scale = 30)

# shapefile_f = "./shapefile/test_polygon.shp"
import gee_time_series
importlib.reload(gee_time_series)
shapefile_f = "./shapefile/dam_location.shp"
shapefile_f = "./shapefile/buffer0.shp"
shapefile_f = "./shapefile/id2_buffer0.shp"
shapefile_f = "./shapefile/id3_buffer0.shp"
df_pad_sentinel10 = gee_time_series.gee_subset_shape(product="COPERNICUS/S1_GRD"      , bands=Sentinel1_bands, start_date=startdate, end_date=enddate, scale=10, shapefile=shapefile_f)
shapefile_f = "./shapefile/buffer1.shp"
shapefile_f = "./shapefile/buffer3.shp"
shapefile_f = "./shapefile/id2_buffer1.shp"
shapefile_f = "./shapefile/id3_buffer1.shp"
df_pad_sentinel11 = gee_time_series.gee_subset_shape(product="COPERNICUS/S1_GRD"      , bands=Sentinel1_bands, start_date=startdate, end_date=enddate, scale=10, shapefile=shapefile_f)
shapefile_f = "./shapefile/buffer2.shp"
shapefile_f = "./shapefile/buffer4.shp"
shapefile_f = "./shapefile/id2_buffer2.shp"
shapefile_f = "./shapefile/id3_buffer2.shp"
df_pad_sentinel12 = gee_time_series.gee_subset_shape(product="COPERNICUS/S1_GRD"      , bands=Sentinel1_bands, start_date=startdate, end_date=enddate, scale=10, shapefile=shapefile_f)

# df_pad_Landsat8 = gee_time_series.gee_subset_shape(product = "LANDSAT/LC08/C02/T1_L2", bands=Landsat_bands  , start_date=startdate, end_date=enddate, scale=30, shapefile=shapefile_f)
# df_pad_Landsat8 = gee_time_series.gee_subset_shape(product = "LANDSAT/LC08/C02/T1_L2", bands=Landsat_bands  , start_date=startdate, end_date=enddate, scale=30, shapefile=shapefile_f)


##*******************************************************************************************************************************************************
## convert patch into images
# get images 
import numpy as np
FILL_VALUE = 65455
import utility
## Plot time series for Sentinel-1 
import os 
Sentinel1_dir = "./Sentinel1_images"
if not os.path.isdir(Sentinel1_dir):
	os.makedirs(Sentinel1_dir)

# image_Sentinel11 =   utility.get_image_fast(df_pad_sentinel11, fields=Sentinel1_bands, dtype=np.float)
importlib.reload(utility)
image_Sentinel10 =   utility.get_image_fast_polygon(df_pad_sentinel10, fields=Sentinel1_bands, dtype=np.float)
image_Sentinel11 =   utility.get_image_fast_polygon(df_pad_sentinel11, fields=Sentinel1_bands, dtype=np.float)
image_Sentinel12 =   utility.get_image_fast_polygon(df_pad_sentinel12, fields=Sentinel1_bands, dtype=np.float)
# np.unique (df_pad_sentinel11['id'])
# utility.plot_time_sereis (df_pad_sentinel11, image_Sentinel11, label="Sentinel-1", figure_dir=Sentinel1_dir, offset=32*100, scale=100,thirdd=1)

importlib.reload(utility)
means0, std0, doys, labels = utility.get_time_sereis_mean_std (df_pad_sentinel10, image_Sentinel10, label="Sentinel-1", figure_dir=Sentinel1_dir)
means1, std1, doys, labels = utility.get_time_sereis_mean_std (df_pad_sentinel11, image_Sentinel11, label="Sentinel-1", figure_dir=Sentinel1_dir)
means2, std2, doys, labels = utility.get_time_sereis_mean_std (df_pad_sentinel12, image_Sentinel12, label="Sentinel-1", figure_dir=Sentinel1_dir)



import matplotlib.pyplot as plt
band=0
# line0, = plt.plot(doys, means0[band,:], c="black")
line1, = plt.plot(doys, means1[band,:], c="blue")
# plt.fill_between(doys, means1[0,:]-std1[0,:], means1[0,:]+std1[0,:])
line2, = plt.plot(doys, means2[band,:], c="red"  )
# plt.fill_between(doys, means2[0,:]-std2[0,:], means2[0,:]+std2[0,:])
plt.xlabel('Day of year in 2021')
plt.ylabel('Backscattering '+Sentinel1_bands[band])
# plt.legend([line0, line1, line2], ['Polygon by Val', 'Close to Dam Polygon', 'Far to Dam Polygon'])
plt.legend([line1, line2], ['Close to Dam Polygon', 'Far to Dam Polygon'])
plt.show()




## Sentinel-1 one for display 
image_Sentinel11_box =   utility.get_image_fast(df_pad_sentinel1_box, fields=Sentinel1_bands, dtype=np.float)
utility.plot_time_sereis (df_pad_sentinel1_box, image_Sentinel11_box, label="Sentinel-1", figure_dir=Sentinel1_dir, offset=32*100, scale=100,thirdd=1)


## Collection 2 
## Plot time series for Landsat-8
Landsat8_dir = "./Landsat8_images"
if not os.path.isdir(Landsat8_dir):
	os.makedirs(Landsat8_dir)

importlib.reload(utility)
image_Landsat = utility.get_image_fast(df_pad_Landsat8, fields=Landsat_bands, dtype=np.uint16)
# image_Landsat = utility.get_image_fast_polygon(df_pad_Landsat8, fields=Landsat_bands, dtype=np.uint16)
np.unique (df_pad_Landsat8['id'])
utility.plot_time_sereis (df_pad_Landsat8, image_Landsat, figure_dir=Landsat8_dir, offset=-0.2*10000, scale=0.0000275*10000)


