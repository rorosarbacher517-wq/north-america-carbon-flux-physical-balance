import requests as r
import numpy as np
# from requests.auth import HTTPBasicAuth
import getpass, pprint, time, os
import json
import appears
import importlib
import geopandas as gpd

output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/images/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable

user = 'binbinfan111'  # Input NASA Earthdata Login Username
password = 'x$v#zH4mMT-Zym$'  # Input NASA Earthdata Login Password

token_response = r.post('{}login'.format(api), auth=(user, password)).json()  # Insert API URL, call login service, provide credentials & return json
# print(prodLayer)
token = token_response['token']  # Save login token to a variable
head = {'Authorization': 'Bearer {}'.format(
    token)}  # Create a header to store token information, needed to submit a request


# # 设置所需的产品
prods = ['HLSL30.020']  # Start a list for products to be requested, beginning with 'HLSL30.020'
prods.append('HLSS30.020')  # Append the HLSS30.020  product to the list of products desired
# print(prods )                       # Print list

# 选择每个产品所需的波段
# L30
layers = [(prods[0], 'B01'), (prods[0], 'B02'), (prods[0], 'B03'), (prods[0], 'B04'),
          (prods[0], 'B05'), (prods[0], 'B06'), (prods[0], 'B07'),
          (prods[0], 'B09'), (prods[0], 'B10'), (prods[0], 'B11'), (prods[0], 'Fmask'),
          (prods[0], 'SZA'), (prods[0], 'SAA'), (prods[0], 'VZA'),
          (prods[0], 'VAA')]  # Create tupled list linking desired product with desired layers

# S30
layers.extend([(prods[1], 'B01'), (prods[1], 'B02'), (prods[1], 'B03'), (prods[1], 'B04'),
               (prods[1], 'B05'), (prods[1], 'B06'), (prods[1], 'B07'), (prods[1], 'B08'),
               (prods[1], 'B8A'), (prods[1], 'B09'), (prods[1], 'B10'), (prods[1], 'B11'),
               (prods[1], 'B12'), (prods[1], 'Fmask'), (prods[0], 'SZA'), (prods[0], 'SAA'),
               (prods[0], 'VZA'),
               (prods[0], 'VAA')])  # Append to tupled list linking desired product with desired layers

# 将所需的波段和对应的产品以layer添加进去
prodLayer = []
for l in layers:
    prodLayer.append({
        "layer": l[1],
        "product": l[0]
    })

projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
projs = {}  # Create an empty dictionary
for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys

shp_path = 'E:/HLS carbon flux/data/sites_263_shp/Sites_263_3km_buffer.geojson'
nps = gpd.read_file(shp_path)  # Read in shapefile as dataframe using geopandas
years = range(2013, 2021)
