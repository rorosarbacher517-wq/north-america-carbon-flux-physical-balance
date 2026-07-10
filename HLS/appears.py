# appears.py 
import requests as r
from requests.auth import HTTPBasicAuth
import getpass, pprint, time, os
import json
import geopandas
import datetime

NUM_PER_TASK = 2
NUM_PER_TASK = 1000
import geopandas as gpd
import requests
import json
import time


##**************************************
## request tasks from polygon shape files
def get_request_polygon_shape_parameter(station_id, prodLayer, projs, geo_json, startDate, endDate):
    task_name = station_id  # User-defined name of the task: 'NPS Vegetation Area' used in example
    task_type = ['point', 'area']  # Type of task, area or point
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

    return task


def get_request_polygon_shape_task1(api, task, head, station_id, station_year_folder):
    task_response = r.post('{}task'.format(api), json=task,
                           headers=head).json()  # Post json to the API task service, return response as json
    # print(task_response)                                                                # Print task response

    # params = {'limit': 2, 'pretty': True}  # Limit API response to 2 most recent entries, return as pretty json
    # tasks_response = r.get('{}task'.format(api), params=params,
    #                        headers=head).json()  # Query task service, setting params and header
    # print(tasks_response)                                                                  # Print tasks response
    # 检查task_response是否包含task_id
    if 'task_id' not in task_response:
        print(f"Error submitting task for {station_id}: {task_response}")  # 输出错误信息

    task_id = task_response['task_id']  # Set task id from request submission
    # status_response = r.get('{}status/{}'.format(api, task_id),
    #                         headers=head).json()  # Call status service with specific task ID & user credentials
    # print(status_response)

    # Ping API until request is complete, then continue to Section 4
    starttime = time.time()
    while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
        print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
        time.sleep(20.0 - ((time.time() - starttime) % 20.0))
    print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])

    bundle = r.get('{}bundle/{}'.format(api, task_id),headers=head).json()  # Call API and return bundle contents for the task_id as json
    print(bundle)  # Print bundle contents
    return bundle


def download_polygon_shape_task1(api, task_id, bundle, head, station_year_folder):
    files = {}  # Create empty dictionary
    for f in bundle['files']: files[f['file_id']] = f['file_name']  # Fill dictionary with file_id as keys and file_name as values
    print(files)

    for f in files:
        dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True,
                   allow_redirects='True')  # Get a stream to the bundle file
        if files[f].endswith('.tif'):
            filename = files[f].split('/')[1]
        else:
            filename = files[f]
        filepath = os.path.join(station_year_folder, filename)  # Create output file path
        if os.path.isfile(filepath):  # 使用 os.path.isfile 检查文件是否存在
            print(f"File {filepath} already exists. Skipping download.")
            continue
        with open(filepath, 'wb') as f:  # Write file to dest dir
            for data in dl.iter_content(chunk_size=8192): f.write(data)
    print('Downloaded files can be found at: {}'.format(station_year_folder))


def get_request_polygon_shape_task2(shape_file, _task, token, fail_sams, name_prefix="", polygon_index=0):
    ls_failed = ['sample_id_start,sample_id_end']
    # shp_gpd = gpd.read_file(shape_file)  # Read the shapefile
    shp_gpd = shape_file  # Read the shapefile
    total_length = len(shp_gpd)

    if polygon_index < 0 or polygon_index >= total_length:
        print("Invalid polygon index. It must be between 0 and {}.".format(total_length - 1))
        return None  # Handle the case of an invalid index

    # Process only the specified polygon
    # ps = {}
    # ps["id"] = '{:05d}'.format(shp_gpd['Site_Id'][polygon_index])  # Assuming there is a unique identifier
    # geometry = shp_gpd['geometry'][polygon_index]
    # ps["polygon"] = geometry.__geo_interface__  # Convert to GeoJSON format

    _task["task_name"] = '{:s}_task_{:03d}_end{:05d}'.format(name_prefix, polygon_index + 1, polygon_index + 1)
    _task["params"]["geo"] = shp_gpd  # Use geo for a single polygon

    # print("...submit task {:3d}\tlastid {:05d}".format(polygon_index + 1, int(ps['id'])))
    time.sleep(30)  # sleep for 30 seconds

    # Submit the task request
    response = requests.post(
        'https://appeears.earthdatacloud.nasa.gov/api/task',
        json=_task,
        headers={'Authorization': 'Bearer {}'.format(token)}
    )

    _status = response.status_code
    for _try in range(5):
        if _status != 202:
            time.sleep(30)  # sleep for 30 seconds
            response = requests.post(
                'https://appeears.earthdatacloud.nasa.gov/api/task',
                json=_task,
                headers={'Authorization': 'Bearer {}'.format(token)}
            )
            _status = response.status_code  # Update status after each retry
        else:
            break

    if _try == 4:
        print('Sample at index %s is not posted successfully' % polygon_index)
        ls_failed.append('%s' % polygon_index)

    with open(fail_sams, 'w') as _fo:
        _fo.write('\n'.join(ls_failed))

    return _task


def download_task_polygon_shape_task2(token, tif_dir, check_frequency=3600, is_download=False, task_str="ARD_C2_n"):
    # Get the list of tasks with the current token
    response = requests.get('https://appeears.earthdatacloud.nasa.gov/api/task',
                            headers={'Authorization': 'Bearer {}'.format(token)})
    task_response = response.json()

    # If is_download is False, loop through tasks
    if not is_download:
        while len(task_response):
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            for _taski in task_response:
                if _taski["status"] != "expired" and task_str in _taski["task_name"]:
                    print("\ttask = {:s}, status = {:s}, created = {:s}, updated = {:s}, completed = {:s}".format(
                    _taski["task_name"], _taski["status"], _taski['created'][:16], _taski['updated'][:16],
                    _taski['completed'][:16] if _taski["status"] == "done" else "Not done"))

                if _taski["status"] == "done":
                    _task_id = _taski["task_id"]
                    _task_name = _taski["task_name"]

                    # Check for files in the bundle
                    response = r.get(
                        'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(_task_id),
                        headers={'Authorization': 'Bearer {0}'.format(token)})

                    bundle_response = response.json()
                    _file_num = 0
                    for _file in bundle_response['files']:
                        _file_id = _file['file_id']
                        _file_name = _file['file_name']
                        # Only download TIFF files

                        if _file_name.endswith('.tif'):
                            _file_num += 1
                            _ff_out = os.path.join(tif_dir, _file_name)
                            if not os.path.exists(_ff_out):
                                print("Starting to download {:s}".format(_ff_out))
                                response = requests.get(
                                    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(_task_id,
                                                                                                         _file_id),
                                    headers={'Authorization': 'Bearer {}'.format(token)},
                                    allow_redirects=True,
                                    stream=True)

                                _status = response.status_code

                                for _try in range(5):
                                    if _status != 200:
                                        time.sleep(5)
                                        response = requests.get(
                                            'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(
                                                _task_id, _file_id),
                                            headers={'Authorization': 'Bearer {}'.format(token)},
                                            allow_redirects=True,
                                            stream=True)
                                        _status = response.status_code  # Update status after each try
                                    else:
                                        break

                                if _try != 4:
                                    os.makedirs(os.path.dirname(_ff_out), exist_ok=True)
                                    with open(_ff_out, 'wb') as f:
                                        for data in response.iter_content(chunk_size=4096):
                                            f.write(data)
                                    print('Downloaded file saved as: {}'.format(_ff_out))
                                else:
                                    print('Failed to download after multiple attempts: {}'.format(_file_name))


                    # delete_task(token, _task_id)
        print('Sleeping for a while to wait for task processing...')
        # time.sleep(check_frequency)
        # # Re-fetch task list after waiting
        # response = requests.get('https://appeears.earthdatacloud.nasa.gov/api/task',
        #                         headers={'Authorization': 'Bearer {}'.format(token)})
        # task_response = response.json()

    return 0

# def download_task_polygon_shape_task2(token, tif_dir, check_frequency=3600, is_download=False, task_str="ARD_C2_n"):
#     while True:
#         response = requests.get(f'https://appeears.earthdatacloud.nasa.gov/api/task', headers={'Authorization': f'Bearer {token}'})
#         if response.status_code != 200:
#             print("Error fetching task list:", response.status_code)
#             time.sleep(check_frequency)
#             continue
#         task_response = response.json()
#         if len(task_response) == 0:
#             print("No tasks available, waiting...")
#             time.sleep(check_frequency)
#             continue
#
#         for task in task_response:
#             task_name = task['task_name']
#             status = task['status']
#             if status != 'done':
#                 print(f"Task {task_name} is still processing. Status: {status}")
#                 continue
#
#             print(f"Task {task_name} is complete. Downloading files...")
#             task_id = task['task_id']
#             response = requests.get(f'https://appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}', headers={'Authorization': f'Bearer {token}'})
#             if response.status_code != 200:
#                 print(f"Error fetching task bundle: {response.status_code}")
#                 continue
#             bundle_response = response.json()
#             for file in bundle_response['files']:
#                 file_name = file['file_name']
#                 if file_name.endswith('.tif'):
#                     file_id = file['file_id']
#                     download_file(token, task_id, file_id, tif_dir)
#
#         print('Sleeping for task check...')
#         time.sleep(check_frequency)

# Function to download files
def download_file(token, task_id, file_id, tif_dir):
    url = f'https://appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}/{file_id}'
    response = requests.get(url, headers={'Authorization': f'Bearer {token}'}, stream=True)
    if response.status_code != 200:
        print(f"Failed to download {file_id}. Status code: {response.status_code}")
        return
    file_path = os.path.join(tif_dir, file_id + '.tif')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)
    print(f"Downloaded {file_id} to {file_path}")

## ****************************************************************************************************
## download task 
def download_task(token,CSV_DIR,check_frequency=600*6,is_download=False,task_str="ARD_C2_n" ):
    response = r.get( 'https://appeears.earthdatacloud.nasa.gov/api/task', headers={'Authorization': 'Bearer {0}'.format(token)})
    task_response = response.json()
    
    # if _taski["status"]!="expired" and _taski["status"]=="processing":
    
    if is_download==False:
        while len(task_response): # this cycle keep trying 
            response = r.get( 'https://appeears.earthdatacloud.nasa.gov/api/task', headers={'Authorization': 'Bearer {0}'.format(token)})
            task_response = response.json()
            now = datetime.datetime.now()   
            print(now.strftime("%Y-%m-%d %H:%M:%S"),end="\n")
            for _taski in task_response:
                month = int(_taski['created'][5:7])
                day = int(_taski['created'][8:10])
                # if _taski["status"]!="expired" and month>=4 and day>=16:
                if _taski["status"]!="expired" and task_str in _taski["task_name"] and _taski["status"]!="error":
                    # Get the current date and time
                    print ("\ttask = {:s}, status = {:s}, created = {:s}, updated = {:s}, completed = {:s}".format (
                        _taski["task_name"], _taski["status"], _taski['created'][:16], _taski['updated'][:16], _taski['completed'][:16] if _taski["status"]=="done" else "Not done" ))
            
            time.sleep(check_frequency)
            # time.sleep(10)
        
            # task_response = response.json()
        
        return 0 
    
    while len(task_response): # this cycle keep trying 
        now = datetime.datetime.now()   
        print(now.strftime("%Y-%m-%d %H:%M:%S"),end="\n")
        for _taski in task_response:
            # print (_task["task_name"])
            # print (_task["status"])
            if _taski["status"]!="expired" and task_str in _taski["task_name"]:
                print ("\ttask = {:s}, status = {:s}, created = {:s}, updated = {:s}, completed = {:s}".format (
                    _taski["task_name"], _taski["status"], _taski['created'][:16], _taski['updated'][:16], _taski['completed'][:16] if _taski["status"]=="done" else "Not done" ))
            
            if _taski["status"] == "done":
                _task_id = _taski["task_id"]
                _task_name = _taski["task_name"]
                # break 
                
                response = r.get(
                    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(_task_id),
                    headers={'Authorization': 'Bearer {0}'.format(token)})
                
                bundle_response = response.json()
                _file_num = 0
                for _file in  bundle_response['files']:
                    _file_id = _file['file_id']
                    _file_name = _file['file_name']
                    if _file_name.endswith('csv'):
                        _file_num += 1
                        _ff_out = os.path.join(CSV_DIR, '%s_%s' % (_task_name, _file_name))
                        if os.path.exists(_ff_out):
                            continue
                        
                        # break 
                        print ("start to download {:s}",_ff_out)
                        response = r.get(
                            'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(_task_id, _file_id),
                            headers={'Authorization': 'Bearer {0}'.format(token)},
                            allow_redirects=True,
                            stream=True)
                        
                        _status = response.status_code
                        for _try in range(5):
                            if _status != 200:
                                time.sleep(5)
                                response = r.get(
                                    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(_task_id, _file_id),
                                    headers={'Authorization': 'Bearer {0}'.format(token)},
                                    allow_redirects=True,
                                    stream=True)
                            else:
                                break
                        
                        if _try != 4:
                            os.makedirs(os.path.dirname(_ff_out), exist_ok=True)
                            with open(_ff_out, 'wb') as f:
                                for data in response.iter_content(chunk_size=4096):
                                    f.write(data)
        
                # if len(os.listdir(os.path.dirname(_ff_out))) = _file_num:
                #     delete_task(token, _task_id)
        
        print ('sleeping for a while to wait task processing')
        time.sleep(check_frequency)
        # time.sleep(10)
        response = r.get( 'https://appeears.earthdatacloud.nasa.gov/api/task', headers={'Authorization': 'Bearer {0}'.format(token)})
        task_response = response.json()


## ****************************************************************************************************
## request tasks from shape files
## shape_file = "./LCMAP_CU_V013_REF/ScienceBase_data_forpeer_review_load/LCMAP_Collection1.3_simple.shp"
## name_prefix = "what"
def request_task_shape(shape_file, _task, token, fail_sams, name_prefix="",NUM_PER_TASK=1000, start_i=0, MAX_TASK=2):
    ls_failed = ['sample_id_start,sample_id_end']
    shp_gpd = geopandas.read_file(shape_file)
    # with open(ff_csv, 'r') as _f_in:
        # _info = _f_in.read().splitlines()
    _task_num = 0
    total_length = len(shp_gpd[''])
    for i in range(start_i, total_length, NUM_PER_TASK):
        _task_num += 1
        _cors = []

        # break csv into multiple tasks each with NUM_PER_TASK points
        for _line in range(i,min(i + NUM_PER_TASK, total_length) ):
            # _l = _line.split(',')
            ps = {}
            # ps["id"] =       (str)(shp_gpd['plotid'][_line])
            # ps["latitude"] =  (str)(shp_gpd['y'][_line] )
            # ps["longitude"] = (str)(shp_gpd['x'][_line] )
            ps["id"       ] =  '{:05d}' .format(shp_gpd['plotid'][_line])
            ps["latitude" ] =  '{:.10f}'.format(shp_gpd['y'][_line] )
            ps["longitude"] =  '{:.10f}'.format(shp_gpd['x'][_line] )
            _cors.append(ps)

        # _task["task_name"] = 'task_%s' % _task_num
        _task["task_name"] = '{:s}_task_{:03d}_end{:05d}'.format(name_prefix,_task_num,_line+1)
        _task["params"]["coordinates"] = _cors
        # break
        print ("...submit task {:3d}\tlastid  {:09d}".format(_task_num,int(_cors[-1]['id']) ) );
        # print (_cors[-1]['id'])
        time.sleep(30) # sleep for 30 seconds
        response = r.post(
            'https://appeears.earthdatacloud.nasa.gov/api/task',
            json=_task,
            headers={'Authorization': 'Bearer {0}'.format(token)})

        _status = response.status_code
        for _try in range(5):
            if _status != 202:
                time.sleep(30) # sleep for 30 seconds
                response = r.post(
                    'https://appeears.earthdatacloud.nasa.gov/api/task',
                    json=_task,
                    headers={'Authorization': 'Bearer {0}'.format(token)})
            else:
                break

        if _try == 4:
            print ('sampels from %s to %s is not posted sucessfully' % (i, min(i + NUM_PER_TASK, total_length)))
            ls_failed.append('%s,%s' % (i, min(i + NUM_PER_TASK, total_length)))

        with open(fail_sams, 'w') as _fo:
            _fo.write('\n'.join(ls_failed))

        if _task_num>=MAX_TASK:
            break
        # if i>1:
            # break

    return _task



    
## ****************************************************************************************************
## request tasks from excel
## ff_csv = "./HaoLiu_lonlat_1000_single_tile_11SMT.csv"
## ff_csv = "./HaoLiu_latlon_1000_multi_tiles.csv"
## lat_index=0;lon_index=1;id_index=-1
def request_task(ff_csv, _task, token, fail_sams, name_prefix="",NUM_PER_TASK=1000,lat_index=2,lon_index=3,id_index=-1):
    ls_failed = ['sample_id_start,sample_id_end']
    with open(ff_csv, 'r') as _f_in:
        _info = _f_in.read().splitlines()
        _task_num = 0
        for i in range(1, len(_info), NUM_PER_TASK):
            _task_num += 1
            _cors = []
            
            # break csv into multiple tasks each with NUM_PER_TASK points
            for _line in _info[i:min(i + NUM_PER_TASK, len(_info))]:
                _l = _line.split(',')
                ps = {}
                if id_index==-1:
                    ps["id"] = str(i)
                else:
                    ps["id"] = _l[id_index]
                
                ps["latitude" ] = _l[lat_index]
                ps["longitude"] = _l[lon_index]
                _cors.append(ps)
            
            # _task["task_name"] = 'task_%s' % _task_num
            _task["task_name"] = '{:s}_task_{:03d}'.format(name_prefix,_task_num)
            _task["params"]["coordinates"] = _cors
            
            print ("...submit task {:3d}".format(_task_num) );
            time.sleep(30) # sleep for 30 seconds
            response = r.post(
                'https://appeears.earthdatacloud.nasa.gov/api/task',
                json=_task,
                headers={'Authorization': 'Bearer {0}'.format(token)})
            
            _status = response.status_code
            for _try in range(5):
                if _status != 202:
                    time.sleep(30) # sleep for 30 seconds
                    response = r.post(
                        'https://appeears.earthdatacloud.nasa.gov/api/task',
                        json=_task,
                        headers={'Authorization': 'Bearer {0}'.format(token)})
                else:
                    break 
            
            if _try == 4:
                print ('sampels from %s to %s is not posted sucessfully' % (i, min(i + NUM_PER_TASK, len(_info))))
                ls_failed.append('%s,%s' % (i, min(i + NUM_PER_TASK, len(_info))))
            
            if i>-1:
                break
                
        with open(fail_sams, 'w') as _fo:
            _fo.write('\n'.join(ls_failed))
            

def generate_parameters_more(is_ARD=0,start_date="01-01-2023",end_date="12-31-2023"):
    layers = '''[{
                    "product": "HLSL30.020",
                    "layer": "B01"
            },{
                    "product": "HLSL30.020",
                    "layer": "B02"
             }]'''

    name = "HLS"
    layers = '''[{
                    "product": "HLSL30.020",
                    "layer": "B01"
            },{
                    "product": "HLSL30.020",
                    "layer": "B02"
            },{
                    "product": "HLSL30.020",
                    "layer": "B03"
            },{
                    "product": "HLSL30.020",
                    "layer": "B04"
            },{
                    "product": "HLSL30.020",
                    "layer": "B05"
            },{
                    "product": "HLSL30.020",
                    "layer": "B06"
            },{
                    "product": "HLSL30.020",
                    "layer": "B07"
            },{
                    "product": "HLSL30.020",
                    "layer": "Fmask"
            },{
                    "product": "HLSS30.020",
                    "layer": "B01"
            },{
                    "product": "HLSS30.020",
                    "layer": "B02"
            },{
                    "product": "HLSS30.020",
                    "layer": "B03"
            },{
                    "product": "HLSS30.020",
                    "layer": "B04"
            },{
                    "product": "HLSS30.020",
                    "layer": "B05"
            },{
                    "product": "HLSS30.020",
                    "layer": "B06"
            },{
                    "product": "HLSS30.020",
                    "layer": "B07"
            },{
                    "product": "HLSS30.020",
                    "layer": "B08"
            },{
                    "product": "HLSS30.020",
                    "layer": "B09"
            },{
                    "product": "HLSS30.020",
                    "layer": "B11"
            },{
                    "product": "HLSS30.020",
                    "layer": "B12"
            },{
                    "product": "HLSS30.020",
                    "layer": "B8A"
            },{
                    "product": "HLSS30.020",
                    "layer": "Fmask"
            }]'''
    
    if is_ARD:
        layers = "["
        for dataset in ["L04.002","L05.002","L07.002"]:
            for band in ["QA_RADSAT", "QA_PIXEL", "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]:
                layers+='{\n                \"product\": \"'
                layers+=dataset
                layers+='\",\n                \"layer\": \"'
                layers+=band
                layers+="\"\n        },"
                # layers.append({"product":dataset,"layer":band})
        
        for dataset in ["L08.002","L09.002"]:
            for band in ["QA_RADSAT", "QA_PIXEL", "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]:
                layers+='{\n                \"product\": \"'
                layers+=dataset
                layers+='\",\n                \"layer\": \"'
                layers+=band
                if dataset=="L09.002" and band=="SR_B7":
                    layers+="\"\n        }]"
                else:
                    layers+="\"\n        },"
                # layers.append({"product":dataset,"layer":band})
        
        layers = str(layers)
        name = "ARD"
                # "layers": {layers},
    ## check this https://stackoverflow.com/questions/64115991/use-in-python-triple-quotes-with-argument-replacement
    _task = '''{{
        "task_type": "point",
        "task_name": "{name}",
        "params":
            {{
                "dates": [{{
                    "startDate": "{start_date}",
                    "endDate": "{end_date}",
                    "recurring": false
                    }}],
                "layers": {layers},
                "coordinates": [{{
                "latitude": "36.6050",
                "longitude": "-118.0622",
                "id": "0"}},
                {{
                    "latitude": "36.6050",
                    "longitude": "60.345",
                    "id": "1"
                }}]
    
        }}
    }}'''
    settings = {'name': name, 'start_date': start_date, 'end_date':end_date, 'layers':layers}
    _task2 = _task.format(**settings)
    return _task2


def generate_parameters():
    _task = '''{
        "task_type": "point",
        "task_name": "HLS",
        "params":
            {
                "dates": [{
                    "startDate": "01-01-2023",
                    "endDate": "12-31-2023",
                    "recurring": false
                    }],
                "layers": [{
                    "product": "HLSL30.020",
                    "layer": "B01"
            },{
                    "product": "HLSL30.020",
                    "layer": "B02"
            },{
                    "product": "HLSL30.020",
                    "layer": "B03"
            },{
                    "product": "HLSL30.020",
                    "layer": "B04"
            },{
                    "product": "HLSL30.020",
                    "layer": "B05"
            },{
                    "product": "HLSL30.020",
                    "layer": "B06"
            },{
                    "product": "HLSL30.020",
                    "layer": "B07"
            },{
                    "product": "HLSL30.020",
                    "layer": "Fmask"
            },{
                    "product": "HLSS30.020",
                    "layer": "B01"
            },{
                    "product": "HLSS30.020",
                    "layer": "B02"
            },{
                    "product": "HLSS30.020",
                    "layer": "B03"
            },{
                    "product": "HLSS30.020",
                    "layer": "B04"
            },{
                    "product": "HLSS30.020",
                    "layer": "B05"
            },{
                    "product": "HLSS30.020",
                    "layer": "B06"
            },{
                    "product": "HLSS30.020",
                    "layer": "B07"
            },{
                    "product": "HLSS30.020",
                    "layer": "B08"
            },{
                    "product": "HLSS30.020",
                    "layer": "B09"
            },{
                    "product": "HLSS30.020",
                    "layer": "B11"
            },{
                    "product": "HLSS30.020",
                    "layer": "B12"
            },{
                    "product": "HLSS30.020",
                    "layer": "B8A"
            },{
                    "product": "HLSS30.020",
                    "layer": "Fmask"
            }],
                "coordinates": [{
                "latitude": "36.6050",
                "longitude": "-118.0622",
                "id": "0"
            },{
                "latitude": "36.6050",
                "longitude": "60.345",
                "id": "1"
            }]

        }
    }
    '''
    return _task


# name = 'john'
# age = '18'
# country = 'us'

# grant = """
# path "{name}-{age}-{country}" {{
    # capabilities = [ "update", "read", "list" ]
# }}
# """
# print(grant)

# template = """There is {name}
# and here is a {{
# an later there is a }}"""
# settings = {'name': "John"}
# print(template.format(**settings))

## delete tasks 
def delete_task(token, _task_id):

    response = r.delete(
        'https://appeears.earthdatacloud.nasa.gov/api/task/{0}'.format(_task_id),
        headers={'Authorization': 'Bearer {0}'.format(token)})
    _status = response.status_code
    for _try in range(5):
        if _status != 204:
            time.sleep(5)
            response = r.delete(
                'https://appeears.earthdatacloud.nasa.gov/api/task/{0}'.format(_task_id),
                headers={'Authorization': 'Bearer {0}'.format(token)})
