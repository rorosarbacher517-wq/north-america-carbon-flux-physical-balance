# v1.6 on Oct 12, Fan changed so that he can download shpfiles
# v1.5 on Oct 11, Hank changed so that he can share
# v1.3 test one tile or multiple tiles and fix data downalod function
# v1.2 test one tile or multiple tiles
# v1.1 on Apr 16 2024 based on Che's codes
import requests as r
import numpy as np
# from requests.auth import HTTPBasicAuth
import getpass, pprint, time, os
import json
import appears
import importlib
import geopandas as gpd

# inDir = 'E:/HLS carbon flux/data/sites_263_shp/'           # IMPORTANT: Update to reflect directory on your OS
# os.chdir(inDir)                                      # Change to working directory
shp_path = "E:/HLS carbon flux/data/Sites data/Ameriflux_site_csv_shp/Ameriflux_site_3.5km.geojson"
output_dir =  'E:/HLS carbon flux/data/Sites data/HLS_Time_series/images/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable

user = 'binbinfan111'      # Input NASA Earthdata Login Username
password = 'x$v#zH4mMT-Zym$'  # Input NASA Earthdata Login Password

token_response = r.post('{}login'.format(api), auth=(user, password)).json() # Insert API URL, call login service, provide credentials & return json
# print(prodLayer)
token = token_response['token']                      # Save login token to a variable
head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request

# # 设置所需的产品
prods = ['HLSL30.020']     # Start a list for products to be requested, beginning with 'HLSL30.020'
prods.append('HLSS30.020')  # Append the HLSS30.020  product to the list of products desired
# print(prods )                       # Print list

# 选择每个产品所需的波段
# L30
layers = [(prods[0], 'B01'), (prods[0], 'B02'), (prods[0], 'B03'), (prods[0], 'B04'),
          (prods[0], 'B05'), (prods[0], 'B06'), (prods[0], 'B07'),
          (prods[0], 'B09'), (prods[0], 'B10'), (prods[0], 'B11'),(prods[0], 'Fmask'),
          (prods[0], 'SZA'), (prods[0], 'SAA'), (prods[0], 'VZA'), (prods[0], 'VAA')]  # Create tupled list linking desired product with desired layers

# S30
layers.extend([(prods[1], 'B01'), (prods[1], 'B02'), (prods[1], 'B03'), (prods[1], 'B04'),
               (prods[1], 'B05'), (prods[1], 'B06'), (prods[1], 'B07'), (prods[1], 'B08'),
               (prods[1], 'B8A'), (prods[1], 'B09'), (prods[1],  'B10'), (prods[1], 'B11'),
              (prods[1], 'B12'), (prods[1], 'Fmask'), (prods[0], 'SZA'), (prods[0], 'SAA'),
               (prods[0], 'VZA'), (prods[0], 'VAA')])  # Append to tupled list linking desired product with desired layers

# 将所需的波段和对应的产品以layer添加进去
prodLayer = []
for l in layers:
    prodLayer.append({
        "layer": l[1],
        "product": l[0]
    })


# nps_gc = nps[nps['Site_Id']=='US-NC4'].to_json() # Extract Grand Canyon NP and set to variable
# nps_gc = json.loads(nps_gc)                      # Convert to json format
#
# # 获取所需要的投影
projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
projs = {}                                  # Create an empty dictionary
for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
# print(list(projs.keys())  )                        # Print dictionary keys

# startDate = '01-01-2017'            # Start of the date range for which to extract data: MM-DD-YYYY
# endDate = '01-05-2017' # end of the date range for which to extract data: MM-DD-YYYY
# years = range(2018, 2019)

# 读取shp文件
nps = gpd.read_file(shp_path) # Read in shapefile as dataframe using geopandas
years = range(2013, 2025)
def get_parameter(station_id, prodLayer,projs,geo_json,startDate, endDate):
    task_name = station_id # User-defined name of the task: 'NPS Vegetation Area' used in example
    task_type = ['point','area']        # Type of task, area or point
    proj = projs['geographic']['Name']  # Set output projection
    outFormat = ['geotiff', 'netcdf4']  # Set output file format type
    startDate = startDate  # Start of the date range for which to extract data: MM-DD-YYYY
    endDate = endDate  # End of the date range for which to extract data: MM-DD-YYYY
    recurring = False  # Specify True for a recurring date range

    # 设置task 任务
    task = {
        'task_type': task_type[1],
        'task_name': task_name,
        'params': {
             'dates': [
             {
                 'startDate': startDate,
                 'endDate': endDate
             }],
             'layers': prodLayer,
             'output': {
                     'format': {
                             'type': outFormat[0]},
                             'projection': proj},
             'geo': geo_json,
        }
    }

    return  task

# def get_parameter(station_id, geo_json, startDate, endDate):
#     # 设置task 任务
#     task = {
#         'task_type': 'area',
#         'task_name': station_id,
#         'params': {
#             'dates': [{'startDate': startDate, 'endDate': endDate}],
#             'layers': prodLayer,
#             'output': {
#                 'format': {'type': 'geotiff'},
#                 'projection': projs['geographic']['Name']
#             },
#             'geo': geo_json
#         }
#     }
#     return task
fail_sams = './failed_post_samples.csv'

from shapely.geometry import mapping  # 用于几何数据转换
#*********************************************
# 依次遍历每个'Site_Id'
for i in range(1,2):
    station_id = nps['Name'][i]
    nps_gc = nps[nps['Name'] == station_id].to_json()
    nps_gc = json.loads(nps_gc)
    # # nps_gc = json.dumps(mapping(row.geometry))  # Convert geometry to GeoJSON
    # nps_gc = json.loads(row.geometry.to_json())
    # 创建输出目录
    station_destDir = os.path.join(output_dir, station_id)
    os.makedirs(station_destDir, exist_ok=True)  # Create the output directory
    years = range(nps['AmeriFlux BASE Start'][i], nps['AmeriFlux BASE End'][i])
    for year in years:
        if 2012 < year < 2025:
            start_date = f"01-01-{year}"
            end_date = f"12-31-{year}"
            # Create directory for each station's yearly data
            station_year_folder = os.path.join(station_destDir, str(year))
            # subset
            os.makedirs(station_year_folder, exist_ok=True)
            print(f"Folder ready: {station_year_folder}")

            task = get_parameter(station_id,prodLayer,projs, nps_gc, start_date, end_date)
            # _task1 = appears.get_request_polygon_shape_parameter(station_id, prodLayer, projs, nps_gc, start_date, end_date)
            task_id, bundle = appears.get_request_polygon_shape_task1(api,task,head,station_id,station_year_folder)
            appears.download_polygon_shape_task1(api, task_id, bundle, head, station_year_folder)

            # task2 = appears.request_task_polygon_shape(nps_gc, task, token, fail_sams, name_prefix="",NUM_PER_TASK=2000, start_i=0, MAX_TASK=2)
            # appears.download_task_polygon_shape_task2(token, station_year_folder, check_frequency=3600, is_download=False, task_str="ARD_C2_n")
            print('Posting have been completed')

# # 依次遍历每个'Site_Id'
# for i in range(101,102):
#     station_id = nps['Site_Id'][i]
#     nps_gc = nps[nps['Site_Id'] == station_id].to_json()
#     nps_gc = json.loads(nps_gc)
#     # # nps_gc = json.dumps(mapping(row.geometry))  # Convert geometry to GeoJSON
#     # nps_gc = json.loads(row.geometry.to_json())
#     # 创建输出目录
#     station_destDir = os.path.join(output_dir, station_id)
#     os.makedirs(station_destDir, exist_ok=True)  # Create the output directory
#
#     for year in years:
#         start_date = f"01-01-{year}"
#         end_date = f"12-31-{year}"
#         # Create directory for each station's yearly data
#         station_year_folder = os.path.join(station_destDir, str(year))
#         # subset
#         os.makedirs(station_year_folder, exist_ok=True)
#         print(f"Folder ready: {station_year_folder}")
#
#         task = get_parameter(station_id,projs, nps_gc, start_date, end_date)
#         # _task1 = appears.get_request_shape_parameter(station_id, prodLayer, projs, nps_gc, start_date, end_date)
#         # task2 = appears.get_request_shape_task(api,task,head,station_id,station_year_folder)
#         # task2 = appears.request_task_polygon_shape(nps_gc, task, token, fail_sams, name_prefix="",
#         #                                            NUM_PER_TASK=1000, start_i=0, MAX_TASK=2)
#         task_response = r.post('{}task'.format(api), json=task,headers=head).json()  # Post json to the API task service, return response as json
#         # print(task_response)                                                                # Print task response
#
#         # params = {'limit': 2, 'pretty': True}  # Limit API response to 2 most recent entries, return as pretty json
#         # tasks_response = r.get('{}task'.format(api), params=params,
#         #                        headers=head).json()  # Query task service, setting params and header
#         # print(tasks_response)                                                                  # Print tasks response
#         # 检查task_response是否包含task_id
#         if 'task_id' not in task_response:
#             print(f"Error submitting task for {station_id}: {task_response}")  # 输出错误信息
#             continue  # 跳过当前循环，继续处理下一个站点
#
#         task_id = task_response['task_id']  # Set task id from request submission
#         # status_response = r.get('{}status/{}'.format(api, task_id),
#         #                         headers=head).json()  # Call status service with specific task ID & user credentials
#         # print(status_response)
#
#         # Ping API until request is complete, then continue to Section 4
#         starttime = time.time()
#         while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
#             print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#             time.sleep(20.0 - ((time.time() - starttime) % 20.0))
#         print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#
#         bundle = r.get('{}bundle/{}'.format(api, task_id),
#                        headers=head).json()  # Call API and return bundle contents for the task_id as json
#         print(bundle)  # Print bundle contents
#
#         files = {}  # Create empty dictionary
#         for f in bundle['files']: files[f['file_id']] = f[
#             'file_name']  # Fill dictionary with file_id as keys and file_name as values
#         print(files)
#
#         for f in files:
#             dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True,
#                        allow_redirects='True')  # Get a stream to the bundle file
#             if files[f].endswith('.tif'):
#                 filename = files[f].split('/')[1]
#             else:
#                 filename = files[f]
#             filepath = os.path.join(station_year_folder, filename)  # Create output file path
#             if os.path.isfile(filepath):  # 使用 os.path.isfile 检查文件是否存在
#                 print(f"File {filepath} already exists. Skipping download.")
#                 continue
#             with open(filepath, 'wb') as f:  # Write file to dest dir
#                 for data in dl.iter_content(chunk_size=8192): f.write(data)
#         print('Downloaded files can be found at: {}'.format(station_year_folder))

# # 依次遍历
# # Iterate through the stations starting from index 200
# for index in range(10, 11):  # Get the station at the current index
#     station_id = nps['Site_Id'][index]
#     station = nps[nps['Site_Id'][index]].to_json()
#     nps_gc = json.loads(station)
#     station_destDir = os.path.join(output_dir, station_id)  # Set up output directory using input directory and task name
#     if not os.path.exists(station_destDir): os.makedirs(station_destDir)  # Create the output directory
#
#     for year in years:
#         start_date = "01-01-{:4d}".format(year)
#         end_date = "01-5-{:4d}".format(year)
#         # Create directory for each station's yearly data
#         station_year_folder = os.path.join(station_destDir, str(year))
#         # subset
#         os.makedirs(station_year_folder, exist_ok=True)
#         print(f"Folder ready: {station_year_folder}")
#
#         task = get_parameter(station_id,startDate,endDate)
#         task_response = r.post('{}task'.format(api), json=task,
#                                headers=head).json()  # Post json to the API task service, return response as json
#         # print(task_response)                                                                # Print task response
#
#         params = {'limit': 2, 'pretty': True}  # Limit API response to 2 most recent entries, return as pretty json
#         tasks_response = r.get('{}task'.format(api), params=params,
#                                headers=head).json()  # Query task service, setting params and header
#         # print(tasks_response)                                                                  # Print tasks response
#
#         task_id = task_response['task_id']  # Set task id from request submission
#         status_response = r.get('{}status/{}'.format(api, task_id),
#                                 headers=head).json()  # Call status service with specific task ID & user credentials
#
#         # Ping API until request is complete, then continue to Section 4
#         starttime = time.time()
#         while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
#             print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#             time.sleep(20.0 - ((time.time() - starttime) % 20.0))
#         # print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
#
#
#         bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()  # Call API and return bundle contents for the task_id as json
#         print(bundle)                                                  # Print bundle contents
#
#         files = {}                                                       # Create empty dictionary
#         for f in bundle['files']: files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values
#         print(files   )
#
#         for f in files:
#             dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
#             if files[f].endswith('.tif'):
#                 filename = files[f].split('/')[1]
#             else:
#                 filename = files[f]
#             filepath = os.path.join(station_year_folder, filename)                                                       # Create output file path
#             with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
#                 for data in dl.iter_content(chunk_size=8192): f.write(data)
#         print('Downloaded files can be found at: {}'.format(station_destDir))


# *********************************************************************************

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
