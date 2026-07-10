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

CSV_DIR = "/weld/gsce_weld_7/c2_appears"
if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)

## CHANGE THIS auth=('sunqing3020590', 'sun3020590') TO YOUR LPDAAC USER name and password 
token_response = r.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=('sunqing3020590', 'sun3020590')).json()
# token_response = r.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=('lwind16', '2017uslD')).json()
token = token_response['token']


fail_sams = 'failed_post_samples.csv'
shape_file = "./LCMAP_CU_V013_REF/ScienceBase_data_forpeer_review_load/LCMAP_Collection1.3_simple.shp"
NUM_PER_TASK=1000
# for year in range(1984,2023):
for year in range(1985,2023):
# for year in range(1984,1985):
    _task1 = json.loads(appears.generate_parameters_more(is_ARD=1,start_date="01-01-{:4d}".format(year),end_date="12-31-{:4d}".format(year) )) # Landsat ARD
    # _task1 = json.loads(appears.generate_parameters_more(is_ARD=0,start_date="01-01-2023",end_date="12-31-2023"))
    importlib.reload(appears) 
    print (_task1)
    time.sleep(20)
    task2 = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
        NUM_PER_TASK=NUM_PER_TASK,start_i=3000, MAX_TASK=1)
    task2 = appears.request_task_polygon_shape(shape_file, _task1, token, fail_sams, name_prefix="", NUM_PER_TASK=1000, start_i=0, MAX_TASK=2)

## ***************************************************************************************************************************************
## testing 
NUM_PER_TASK=250
NUM_PER_TASK=1
# for year in range(1984,2023):
# for year in range(1986,2023):
for year in range(1985,1986):
    _task1 = json.loads(appears.generate_parameters_more(is_ARD=1,start_date="01-01-{:4d}".format(year),end_date="12-31-{:4d}".format(year) )) # Landsat ARD
    # _task1 = json.loads(appears.generate_parameters_more(is_ARD=0,start_date="01-01-2023",end_date="12-31-2023"))
    importlib.reload(appears) 
    print (_task1)
    time.sleep(20)
    task_t = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="5ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
        NUM_PER_TASK=NUM_PER_TASK,start_i=35, MAX_TASK=1)


# task_t["params"]["coordinates"]
# >>> task_t["params"]["coordinates"] - trouble 
# [{'id': '00036', 'latitude': '44.7107130684', 'longitude': '-97.7272890128'}]


NUM_PER_TASK=250
NUM_PER_TASK=1
years_array = np.concatenate(( np.array(range(1986,1992)),np.array([1993,1995,1996,1998,1999,2000,2003,2004,2005,2009]),np.array(range(2013,2023)), ))
# for year in range(1984,2023):
# for year in range(1986,2023):
# for year in range(1984,1985):
for year in years_array:
    _task1 = json.loads(appears.generate_parameters_more(is_ARD=1,start_date="01-01-{:4d}".format(year),end_date="12-31-{:4d}".format(year) )) # Landsat ARD
    # _task1 = json.loads(appears.generate_parameters_more(is_ARD=0,start_date="01-01-2023",end_date="12-31-2023"))
    importlib.reload(appears) 
    print (_task1)
    time.sleep(20)
    task2 = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="5ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
        NUM_PER_TASK=35 ,start_i=0, MAX_TASK=1)
    print (_task1)
    time.sleep(20)
    task2 = appears.request_task_shape(shape_file, _task1, token, fail_sams, name_prefix="5ARD_C2_n{:04d}_y{:4d}".format(NUM_PER_TASK,year),
        NUM_PER_TASK=964,start_i=36, MAX_TASK=1)
## ***************************************************************************************************************************************



# ff_csv = "./HaoLiu_lonlat_1000_single_tile_11SMT.csv"
# appears.request_task(ff_csv, _task1, token, fail_sams, name_prefix="Singl_tile",NUM_PER_TASK=100,lat_index=0,lon_index=1)

# ff_csv = "./HaoLiu_latlon_1000_multi_tiles.csv"
# appears.request_task(ff_csv, _task1, token, fail_sams, name_prefix="Multi_tile",NUM_PER_TASK=100,lat_index=1,lon_index=2)

importlib.reload(appears) 
# appears.download_task(token,CSV_DIR,is_download=False,task_str="ARD_C2_n" )
appears.download_task(token,CSV_DIR,is_download=True,task_str="ARD_C2_n" )
print ('Posting have been completed')

## sleep for a while before request downloading 
# time.sleep(3600*24)

# token_response = r.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=('skylan', '646583hkHK')).json()
# token = token_response['token']
# download_task(token, dd_out)
# print ('Downloading have been completed')

# response = r.post(
    # 'https://appeears.earthdatacloud.nasa.gov/api/logout',
    # headers={'Authorization': 'Bearer {0}'.format(token)})
