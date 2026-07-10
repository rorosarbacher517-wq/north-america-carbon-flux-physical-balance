# v1.6 on Oct 12, Fan changed so that he can download shpfiles
# v1.5 on Oct 11, Hank changed so that he can share
# v1.3 test one tile or multiple tiles and fix data downalod function
# v1.2 test one tile or multiple tiles
# v1.1 on Apr 16 2024 based on Che's codes
import os
import time
import json
import requests
import geopandas as gpd
from shapely.geometry import mapping  # 用于几何数据转换


# API 登录并获取token
def login(api, user, password):
    response = requests.post(f'{api}login', auth=(user, password)).json()
    return response['token']


# 创建请求头
def create_headers(token):
    return {'Authorization': f'Bearer {token}'}


# 获取所需的投影
def get_projections(api):
    projections = requests.get(f'{api}spatial/proj').json()
    return {p['Name']: p for p in projections}


# 获取任务参数
def get_task_parameters(station_id, layers, geo_json, start_date, end_date, projection):
    return {
        'task_type': 'area',
        'task_name': station_id,
        'params': {
            'dates': [{'startDate': start_date, 'endDate': end_date}],
            'layers': layers,
            'output': {
                'format': {'type': 'geotiff'},
                'projection': projection
            },
            'geo': geo_json
        }
    }


# 处理任务
def handle_task(api, task, headers):
    task_response = requests.post(f'{api}task', json=task, headers=headers).json()
    if 'task_id' not in task_response:
        print(f"Error submitting task: {task_response}")
        return None, None

    task_id = task_response['task_id']
    # Ping API until request is complete
    while True:
        status_response = requests.get(f'{api}task/{task_id}', headers=headers).json()
        print(f'Task status: {status_response["status"]}')
        if status_response['status'] == 'done':
            break
        time.sleep(20)

    # 返回 task_id 和 bundle
    return task_id, requests.get(f'{api}bundle/{task_id}', headers=headers).json()


# 下载文件
def download_files(bundle, output_dir, task_id, existing_files):
    files = {f['file_id']: f['file_name'] for f in bundle['files']}

    for f_id, file_name in files.items():
        # 获取文件路径并检查是否存在
        filepath = os.path.join(output_dir, file_name.split('/')[-1])
        if filepath in existing_files:
            print(f"File {filepath} already exists. Skipping download.")
            continue

        print(f"Downloading {file_name}...")
        response = requests.get(f'{api}bundle/{task_id}/{f_id}', headers=headers, stream=True)

        with open(filepath, 'wb') as f:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
        print(f'Saved {filepath}')


# 设置路径和登录信息
shp_path = 'E:/HLS carbon flux/data/sites_263_shp/Sites_263_2km_buffer.geojson'
output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/images/'
os.makedirs(output_dir, exist_ok=True)
api = 'https://appeears.earthdatacloud.nasa.gov/api/'

user = '2021206190012@whu.edu.cn'
password = '68867FB666lll@'


# 登录
token = login(api, user, password)
headers = create_headers(token)
projections = get_projections(api)

# 设置所需产品和层
products = ['HLSL30.020', 'HLSS30.020']
layers = [
             (product, band) for product in products for band in
             ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask']
         ] + [(products[1], band) for band in ['B01', 'B02', 'B03', 'B04', 'B8A', 'B11', 'B12', 'Fmask']]
prodLayer = [{'layer': l[1], 'product': l[0]} for l in layers]

# 读取shp文件
nps = gpd.read_file(shp_path)
years = range(2013, 2021)

# 遍历每个站点
# for index, row in nps.iterrows():
for i in range(10,20):
    station_id = nps['Site_Id'][i]
    nps_gc = nps[nps['Site_Id'] == station_id].to_json()
    nps_gc = json.loads(nps_gc)

    station_dest_dir = os.path.join(output_dir, station_id)
    os.makedirs(station_dest_dir, exist_ok=True)

    for year in years:
        start_date = f"01-01-{year}"
        end_date = f"12-31-{year}"
        station_year_folder = os.path.join(station_dest_dir, str(year))
        os.makedirs(station_year_folder, exist_ok=True)
        print(f"Folder ready: {station_year_folder}")

        task = get_task_parameters(station_id, prodLayer, nps_gc, start_date, end_date,
                                   projections['geographic']['Name'])

        task_id, bundle = handle_task(api, task, headers)
        if task_id and bundle:
            # 获取已存在文件的列表
            existing_files = os.listdir(station_year_folder)
            full_existing_files = [os.path.join(station_year_folder, f) for f in existing_files]
            download_files(bundle, station_year_folder, task_id, full_existing_files)

# token_response = r.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
#
# # # 设置所需的产品
# prods = ['HLSL30.020']     # Start a list for products to be requested, beginning with 'HLSL30.020'
# prods.append('HLSS30.020')  # Append the HLSS30.020  product to the list of products desired
# # print(prods )                       # Print list
#
# # 选择每个产品所需的波段
# # L30
# layers = [(prods[0], 'B01'), (prods[0], 'B02'), (prods[0], 'B03'), (prods[0], 'B04'),
#           (prods[0], 'B05'), (prods[0], 'B06'), (prods[0], 'B07'), (prods[0], 'Fmask')]  # Create tupled list linking desired product with desired layers
#
# # S30
# layers.extend([(prods[1], 'B01'), (prods[1], 'B02'), (prods[1], 'B03'), (prods[1], 'B04'),
#                (prods[1], 'B8A'), (prods[1], 'B11'), (prods[1], 'B12'), (prods[1], 'Fmask')])  # Append to tupled list linking desired product with desired layers
#
# # 将所需的波段和对应的产品以layer添加进去
# prodLayer = []
# for l in layers:
#     prodLayer.append({
#         "layer": l[1],
#         "product": l[0]
#     })
#
# # print(prodLayer)
# token = token_response['token']                      # Save login token to a variable
# head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request
#
# # 读取shp文件
# nps = gpd.read_file(shp_path) # Read in shapefile as dataframe using geopandas
# # print(nps.head())                                                # Print first few lines of dataframe
#
# nps_gc = nps[nps['Site_Id']=='US-NC4'].to_json() # Extract Grand Canyon NP and set to variable
# nps_gc = json.loads(nps_gc)                      # Convert to json format
#
# # 获取所需要的投影
# projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
# projs = {}                                  # Create an empty dictionary
# for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
# # print(list(projs.keys())  )                        # Print dictionary keys
#
# startDate = '01-01-2017'            # Start of the date range for which to extract data: MM-DD-YYYY
# endDate = '01-05-2017' # end of the date range for which to extract data: MM-DD-YYYY
# years = range(2018, 2019)
#
# def get_parameter(station_id, geo_json, startDate, endDate):
#     task_name = station_id # User-defined name of the task: 'NPS Vegetation Area' used in example
#     task_type = ['point','area']        # Type of task, area or point
#     proj = projs['geographic']['Name']  # Set output projection
#     outFormat = ['geotiff', 'netcdf4']  # Set output file format type
#     startDate = startDate  # Start of the date range for which to extract data: MM-DD-YYYY
#     endDate = endDate  # End of the date range for which to extract data: MM-DD-YYYY
#     recurring = False  # Specify True for a recurring date range
#
#     # 设置task 任务
#     task = {
#         'task_type': task_type[1],
#         'task_name': task_name,
#         'params': {
#              'dates': [
#              {
#                  'startDate': startDate,
#                  'endDate': endDate
#              }],
#              'layers': prodLayer,
#              'output': {
#                      'format': {
#                              'type': outFormat[0]},
#                              'projection': proj},
#              'geo': geo_json,
#         }
#     }
#
#     return  task
#
#
# # *********************************************************************************
#
# token_response = r.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
# del user, password
# # print(token_response )
#
# product_response = r.get('{}product'.format(api)).json()                         # request all products in the product service
# # print('AρρEEARS currently supports {} products.'.format(len(product_response)))  # Print no. products available in AppEEARS
#
# products = {p['ProductAndVersion']: p for p in product_response} # Create a dictionary indexed by product name & version
# # print(products['HLSL30.020'])
# # products['HLSL30.020','HLSS30.020']
#
# prodNames = {p['ProductAndVersion'] for p in product_response} # Make list of all products (including version)
# for p in prodNames:                                            # Make for loop to search list of products 'Description' for a keyword
#     if 'Land Surface Reflectance' in products[p]['Description']:
#         pprint.pprint(products[p])                             # Print info for each product containing LAI in its description
#
#
# prods = ['HLSL30.020']     # Start a list for products to be requested, beginning with MCD15A3H.006
# prods.append('HLSS30.020')  # Append the MOD11A2.061 8 day LST product to the list of products desired
# # print(prods )                       # Print list
#
# lst_response = r.get('{}product/{}'.format(api, prods[0])).json()  # Request layers for the 2nd product (index 1) in the list: MOD11A2.061
# print(list(lst_response.keys()))
#
# lai_response = r.get('{}product/{}'.format(api, prods[1])).json()  # Request layers for the 1st product (index 0) in the list: MCD15A3H.006
# print(list(lai_response.keys()))                                        # Print the LAI layer names
#
#
# layers = [(prods[0],'B01'),(prods[0],'B02')]  # Create tupled list linking desired product with desired layers
# # lai_response['Lai_500m']['Description']  # Make sure the correct layer is requested
# layers.append((prods[1],'B03')) # Append to tupled list linking desired product with desired layers
#
# prodLayer = []
# for l in layers:
#     prodLayer.append({
#             "layer": l[1],
#             "product": l[0]
#           })
# # print(prodLayer)
#
# token = token_response['token']                      # Save login token to a variable
# head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request
#
# nps = gpd.read_file(shp_path) # Read in shapefile as dataframe using geopandas
# # print(nps.head())                                                # Print first few lines of dataframe
#
# nps_gc = nps[nps['Site_Id']=='US-NC4'].to_json() # Extract Grand Canyon NP and set to variable
# nps_gc = json.loads(nps_gc)                                            # Convert to json format
#
# projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
# # print(projections )                                              # Print projections and information
#
# projs = {}                                  # Create an empty dictionary
# for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
# # print(list(projs.keys())  )                        # Print dictionary keys
#
# projs['geographic'] # 'native'
#
# task_name = 'HLS' # User-defined name of the task: 'NPS Vegetation Area' used in example
#
# task_type = ['point','area']        # Type of task, area or point
# proj = projs['geographic']['Name']  # Set output projection
# outFormat = ['geotiff', 'netcdf4']  # Set output file format type
# startDate = '07-01-2017'            # Start of the date range for which to extract data: MM-DD-YYYY
# endDate = '07-05-2017'              # End of the date range for which to extract data: MM-DD-YYYY
# recurring = False                   # Specify True for a recurring date range
# #yearRange = [2000,2016]            # if recurring = True, set yearRange, change start/end date to MM-DD
#
# task = {
#     'task_type': task_type[1],
#     'task_name': task_name,
#     'params': {
#          'dates': [
#          {
#              'startDate': startDate,
#              'endDate': endDate
#          }],
#          'layers': prodLayer,
#          'output': {
#                  'format': {
#                          'type': outFormat[0]},
#                          'projection': proj},
#          'geo': nps_gc,
#     }
# }
#
# task_response = r.post('{}task'.format(api), json=task, headers=head).json()  # Post json to the API task service, return response as json
# # print(task_response)                                                                # Print task response
#
# params = {'limit': 2, 'pretty': True} # Limit API response to 2 most recent entries, return as pretty json
# tasks_response = r.get('{}task'.format(api), params=params, headers=head).json() # Query task service, setting params and header
# # print(tasks_response)                                                                  # Print tasks response
#
# task_id = task_response['task_id']                                               # Set task id from request submission
# status_response = r.get('{}status/{}'.format(api, task_id), headers=head).json() # Call status service with specific task ID & user credentials
# # print(status_response)
#
# # Ping API until request is complete, then continue to Section 4
# starttime = time.time()
# while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
#     print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#     time.sleep(20.0 - ((time.time() - starttime) % 20.0))
# # print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#
# destDir = os.path.join(output_dir, task_name)                # Set up output directory using input directory and task name
# if not os.path.exists(destDir):os.makedirs(destDir)     # Create the output directory
#
#
# bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()  # Call API and return bundle contents for the task_id as json
# print(bundle)                                                  # Print bundle contents
#
# files = {}                                                       # Create empty dictionary
# for f in bundle['files']: files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values
# print(files   )
#
# for f in files:
#     dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
#     if files[f].endswith('.tif'):
#         filename = files[f].split('/')[1]
#     else:
#         filename = files[f]
#     filepath = os.path.join(destDir, filename)                                                       # Create output file path
#     with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
#         for data in dl.iter_content(chunk_size=8192): f.write(data)
# print('Downloaded files can be found at: {}'.format(destDir))
# #


# # 存放images的根目录
# output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/images/'
# if not os.path.isdir(output_dir):
#     os.makedirs(output_dir)
#
# # CSV_DIR = "/weld/gsce_weld_7/c2_appears"
# # if not os.path.isdir(CSV_DIR):
# #     os.makedirs(CSV_DIR)
#
#
# api = 'https://appeears.earthdatacloud.nasa.gov/api/'
# token_response = r.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
#
#
#
#
#
# ## CHANGE THIS auth=('sunqing3020590', 'sun3020590') TO YOUR LPDAAC USER name and password
# token_response = r.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(user, password)).json()
# token = token_response['token']
# product_response = r.get('{}product'.format(api)).json()                         # request all products in the product service
# print('AρρEEARS currently supports {} products.'.format(len(product_response)))  # Print no. products available in AppEEARS
# del user, password                                                           # Remove user and password information
# token_response
#
# fail_sams = 'E:/HLS carbon flux/data/HLS_Time_series/failed_post_samples.csv'
# # shape_file = "./LCMAP_CU_V013_REF/ScienceBase_data_forpeer_review_load/LCMAP_Collection1.3_simple.shp"
# shape_file = "E:/HLS carbon flux/data/sites_263_shp/Sites_263_3km_buffer.shp"
#
# ## ***************************************************************************************************************************************
# # ## testing
# NUM_PER_TASK=250
# NUM_PER_TASK=1
# # for year in range(1984,2023):
# # for year in range(1986,2023):
# for year in range(2013,2022):
#     # _task1 = json.loads(appears.generate_parameters_more(is_ARD=1,start_date="01-01-{:4d}".format(year),end_date="12-31-{:4d}".format(year) )) # Landsat ARD
#     _task1 = json.loads(appears.generate_parameters_more(is_ARD=0,start_date="01-01-{:4d}",end_date="12-31-{:4d}"))
#     importlib.reload(appears)
#     print (_task1)
#     time.sleep(20)
#     task_t = appears.request_task_polygon_shape(shape_file, token)
#     print(task_t)

#
# # task_t["params"]["coordinates"]
# # >>> task_t["params"]["coordinates"] - trouble
# # [{'id': '00036', 'latitude': '44.7107130684', 'longitude': '-97.7272890128'}]
#
#
# NUM_PER_TASK=250
# NUM_PER_TASK=1
# years_array = np.concatenate(( np.array(range(1986,1992)),np.array([1993,1995,1996,1998,1999,2000,2003,2004,2005,2009]),np.array(range(2013,2023)), ))
# # for year in range(1984,2023):
# # for year in range(1986,2023):
# # for year in range(1984,1985):
# for year in years_array:
#     _task1 = json.loads(appears.generate_parameters_more(is_ARD=1,start_date="01-01-{:4d}".format(year),end_date="12-31-{:4d}".format(year) )) # Landsat ARD
#     # _task1 = json.loads(appears.generate_parameters_more(is_ARD=0,start_date="01-01-2023",end_date="12-31-2023"))
#     importlib.reload(appears)
#     print (_task1)
#     time.sleep(20)
#     task2 = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="5ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
#         NUM_PER_TASK=35 ,start_i=0, MAX_TASK=1)
#     print (_task1)
#     time.sleep(20)
#     task2 = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="5ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
#         NUM_PER_TASK=964,start_i=36, MAX_TASK=1)
# ## ***************************************************************************************************************************************
#
#
#
# # ff_csv = "./HaoLiu_lonlat_1000_single_tile_11SMT.csv"
# # appears.request_task(ff_csv, _task1, token, fail_sams, name_prefix="Singl_tile",NUM_PER_TASK=100,lat_index=0,lon_index=1)
#
# # ff_csv = "./HaoLiu_latlon_1000_multi_tiles.csv"
# # appears.request_task(ff_csv, _task1, token, fail_sams, name_prefix="Multi_tile",NUM_PER_TASK=100,lat_index=1,lon_index=2)
#
# importlib.reload(appears)
# # appears.download_task(token,CSV_DIR,is_download=False,task_str="ARD_C2_n" )
# appears.download_task(token,CSV_DIR,is_download=True,task_str="ARD_C2_n" )

# print ('Posting have been completed')

## sleep for a while before request downloading
# time.sleep(3600*24)

# token_response = r.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=('skylan', '646583hkHK')).json()
# token = token_response['token']
# download_task(token, dd_out)
# print ('Downloading have been completed')

# response = r.post(
    # 'https://appeears.earthdatacloud.nasa.gov/api/logout',
    # headers={'Authorization': 'Bearer {0}'.format(token)})

#
# # 第一步：获取 Token
# username = "binbinfan111"
# password = "x$v#zH4mMT-Zym$"
# url_auth = "https://appeears.earthdatacloud.nasa.gov/api/auth"
# response = requests.post(url_auth, auth=(username, password))
# token = response.json()['token']
#
# # 第二步：定义任务
# url_task = "https://appeears.earthdatacloud.nasa.gov/api/task"
# headers = {
#     'Authorization': f'Bearer {token}',
#     'Content-Type': 'application/json'
# }
#
# # 根据需求定义请求体，以下为示例
# task_payload = {
#     "params": {
#         "geo": json.loads(open("output.geojson").read()),  # 如果需要 GeoJSON
#         "layer": "HLSL30",  # 举例使用 HLS 层
#         "startDate": "2017-01-01",
#         "endDate": "2017-12-31"
#     }
# }
#
# # 提交任务
# response_task = requests.post(url_task, headers=headers, json=task_payload)
# task_id = response_task.json()['task_id']
#
# # 第三步：检查任务状态并下载结果
# url_task_status = f"https://appeears.earthdatacloud.nasa.gov/api/task/{task_id}"
# while True:
#     status_response = requests.get(url_task_status, headers=headers)
#     status = status_response.json()['params']['status']
#
#     if status == 'done':
#         print("任务已完成。开始下载文件。")
#         # 在这里添加代码以下载 HLS 文件，例如：
#         # urls = status_response.json()['downloadUrls']
#         break
#     elif status == 'error':
#         print("任务出现错误。")
#         break
#
#     print("任务正在处理中...")

import subprocess
# Authenticate using Earthaccess
# earthaccess.login(strategy="netrc")

# user = getpass.getpass(prompt = 'Enter NASA Earthdata Login Username: ')      # Input NASA Earthdata Login Username
# password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ')  # Input NASA Earthdata Login Password
# # Set input directory, change working directory
#
# api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable
# token_response = requests.post('{}login'.format(api), auth=(binbinfan111, password)).json() # Insert API URL, call login service, provide credentials & return json
# del user, password                                                           # Remove user and password information
# token_response
# task_name = input('Enter a Task Name: ') # User-defined name of the task: 'NPS Vegetation Area' used in example
#
# task_type = ['point','area']        # Type of task, area or point
# proj = projs['geographic']['Name']  # Set output projection
# outFormat = ['geotiff', 'netcdf4']  # Set output file format type
# startDate = '07-01-2017'            # Start of the date range for which to extract data: MM-DD-YYYY
# endDate = '07-31-2017'              # End of the date range for which to extract data: MM-DD-YYYY
# recurring = False                   # Specify True for a recurring date range
# #yearRange = [2000,2016]            # if recurring = True, set yearRange, change start/end date to MM-DD
# task = {
#     'task_type': task_type[1],
#     'task_name': task_name,
#     'params': {
#          'dates': [
#          {
#              'startDate': startDate,
#              'endDate': endDate
#          }],
#          'layers': prodLayer,
#          'output': {
#                  'format': {
#                          'type': outFormat[0]},
#                          'projection': proj},
#          'geo': nps_gc,
#     }
# }
#
# token = token_response['token']                      # Save login token to a variable
# head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request
# files = {}
# # Create empty dictionary
# destDir = os.path.join(inDir, task_name)                # Set up output directory using input directory and task name
# if not os.path.exists(destDir):os.makedirs(destDir)     # Create the output directory
#
# bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()
# for f in bundle['files']: files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values
#
# for f in files:
#     dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
#     if files[f].endswith('.tif'):
#         filename = files[f].split('/')[1]
#     else:
#         filename = files[f]
#     filepath = os.path.join(destDir, filename)                                                       # Create output file path
#     with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
#         for data in dl.iter_content(chunk_size=8192): f.write(data)
# print('Downloaded files can be found at: {}'.format(destDir))

# # 定义下载列表文件路径
# download_list_path = 'E:/HLS carbon flux/data/HLS_Time_series/download_list/'
# output_dir = 'E:/HLS carbon flux/data/HLS_Time_series/images/'
#
#
# def download_file_with_retry(url, retries=3, delay=5):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#         # 添加其他需要的头，例如认证令牌
#     }
#     for attempt in range(retries):
#         try:
#             response = requests.get(url, headers=headers)
#             response.raise_for_status()  # 如果请求返回码不是200，将引发HTTPError
#             return response.content
#         except requests.RequestException as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             time.sleep(delay)  # 等待一段时间后重试
#     return None  # 如果所有重试都失败
#
#
# # 遍历下载列表中的文件
# # 遍历下载列表中的文件
# for filename in os.listdir(download_list_path):
#     site_name = filename[:6]
#     year_start = 6
#     year_end = filename.find('-', year_start)
#
#     if year_end != -1:
#         year = filename[year_start:year_end]
#         station_year_folder = os.path.join(output_dir, site_name, str(year))
#         os.makedirs(station_year_folder, exist_ok=True)
#
#         print(f'准备下载文件：{filename}')
#
#         # 读取链接文件并下载 .tif 文件
#         link_file = os.path.join(download_list_path, filename)
#         with open(link_file, 'r') as file:
#             urls = file.readlines()
#
#         for url in urls:
#             url = url.strip()  # 去掉多余的空白符
#             if url:  # 确保 URL 不为空
#                 tifname = url.split('/')[-1]  # 提取文件名
#                 file_path = os.path.join(output_dir, tifname)  # 生成完整的文件路径
#
#                 # 使用 requests 下载文件
#                 try:
#                     print(f"Downloading: {url}")
#                     response = requests.get(url)
#                     response.raise_for_status()  # 如果请求返回码不是200，将引发HTTPError
#                     with open(file_path, 'wb') as f:
#                         f.write(response.content)
#                     print(f'Successfully downloaded: {tifname}')
#                 except requests.HTTPError as e:
#                     print(f'Failed to download {url}, Error: {e}')

        # # 下载每个 TIFF 文件
        # for url in urls:
        #     url = url.strip()  # 移除多余的空格
        #     content = download_file_with_retry(url)
        #     if content is not None:
        #         tifname = url.split('/')[-1]
        #         file_path = os.path.join(station_year_folder, tifname)
        #
        #         with open(file_path, 'wb') as f:
        #             f.write(content)
        #         print(f'Successfully downloaded: {tifname}')
        #     else:
        #         print(f'Failed to download {url} after retries.')


    # for url in urls:
    #     url = url.strip()  # 去掉首尾的空白符
    #     if url:  # 确保 url 不为空
    #         # 提取文件名（最后一个 / 后面的字符）
    #         tifname = url.split('/')[-1]
    #         # 定义下载文件的完整路径
    #         file_path = os.path.join(station_year_folder, tifname)
    #
    #         # 下载文件
    #         print(f"正在下载: {url}")
    #         response = requests.get(url)
    #
    #         if response.status_code == 200:
    #             with open(file_path, 'wb') as output_file:
    #                 output_file.write(response.content)
    #             print(f"下载成功: {file_path}")
    #         else:
    #             print(f"下载失败: {url}，状态码: {response.status_code}")


