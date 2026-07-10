import os
import json
import requests as r
import geopandas as gpd
import time
from datetime import datetime

# 定义 API 端点和 NASA Earthdata 的用户凭据
api = 'https://appeears.earthdatacloud.nasa.gov/api/'
user = 'binbinfan111'  # 输入 NASA Earthdata 登录用户名
password = 'x$v#zH4mMT-Zym$'  # 输入 NASA Earthdata 登录密码

# 从 API 获取身份验证令牌
token_response = r.post(f'{api}login', auth=(user, password))
print(token_response.status_code)  # 检查请求的状态码
if token_response.status_code == 200:
    token = token_response.json()['token']  # 成功获取后获取令牌
    head = {'Authorization': f'Bearer {token}'}  # 设置请求头，包含令牌，用于后续请求
else:
    print("错误: 认证失败。")  # 如果认证失败，打印错误信息
    exit()

# 指定将用于数据提取的产品层
prods = ['HLSL30.020', 'HLSS30.020']
layer = [
    (prods[0], 'B01'), (prods[0], 'B02'), (prods[0], 'B03'),
    (prods[0], 'B04'), (prods[0], 'B05'), (prods[0], 'B06'),
    (prods[0], 'B07'), (prods[0], 'B09'), (prods[0], 'B10'),
    (prods[0], 'B11'), (prods[0], 'Fmask'), (prods[0], 'SZA'),
    (prods[0], 'SAA'), (prods[0], 'VZA'), (prods[0], 'VAA')
]
layer.extend([
    (prods[1], 'B01'), (prods[1], 'B02'), (prods[1], 'B03'),
    (prods[1], 'B04'), (prods[1], 'B05'), (prods[1], 'B06'),
    (prods[1], 'B07'), (prods[1], 'B08'), (prods[1], 'B8A'),
    (prods[1], 'B09'), (prods[1], 'B10'), (prods[1], 'B11'),
    (prods[1], 'B12'), (prods[1], 'Fmask'), (prods[0], 'SZA'),
    (prods[0], 'SAA'), (prods[0], 'VZA'), (prods[0], 'VAA')
])

# 创建一个产品层字典，用于 API 请求
prodLayer = [{'layer': l[1], 'product': l[0]} for l in layer]
# 从 API 获取可用的空间投影
projections = r.get('{}spatial/proj'.format(api)).json()
# 创建一个字典来存储投影名称和数据
projs = {}
for p in projections: projs[p['Name']] = p  # 用投影名称作为键填充字典

# 读取包含站点数据的 GeoJSON 文件
shp_path = "E:/HLS carbon flux/data/Sites data/Ameriflux_site_csv_shp/Ameriflux_site_3km.geojson"
nps = gpd.read_file(shp_path)  # 将 GeoJSON 加载为 GeoDataFrame

# 定义下载数据的输出目录
output_dir = 'E:/HLS carbon flux/data/Sites data/HLS_Time_series/images/'
if not os.path.isdir(output_dir):  # 检查目录是否存在；如果不存在，则创建
    os.makedirs(output_dir)

# 获取请求每个站点的参数的函数
def get_parameter(station_id, prodLayer, projs, geo_json, startDate, endDate):
    task_name = station_id  # 根据站点 ID 定义任务名称
    task_type = ['point', 'area']  # 任务类型：点或区域
    proj = projs['geographic']['Name']  # 设置所需的输出投影
    outFormat = ['geotiff', 'netcdf4']  # 定义输出文件格式
    # 创建包含提取详细信息的任务
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
    return task  # 返回构造的任务

def get_request_polygon_shape_task1(api, task, head, station_id):
    task_response = r.post('{}task'.format(api), json=task,
                           headers=head).json()  # 提交任务并接收响应

    # 检查响应是否包含任务 ID；否则打印错误信息
    if 'task_id' not in task_response:
        print(f"提交 {station_id} 的任务时出错: {task_response}")
        return None, None  # 如果出错，就返回 None

    task_id = task_response['task_id']  # 从 API 响应中获取任务 ID

    # 轮询 API 直到任务状态为 'done'
    starttime = time.time()
    while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
        print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])  # 打印当前任务状态
        time.sleep(20.0 - ((time.time() - starttime) % 20.0))  # 延迟以避免过于频繁的轮询
    print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])  # 最终状态打印

    # 获取与任务相关的文件包
    bundle = r.get('{}bundle/{}'.format(api, task_id),
                   headers=head).json()  # 返回与任务 ID 相关的文件内容
    print(bundle)  # 打印文件包的内容以确认

    return task_id, bundle  # 返回任务 ID 和文件包以便下载


# 下载文件的函数，同时包括重试逻辑以增强稳健性
def download_polygon_shape_task1(api, task_id, bundle, head, station_year_folder, max_retries=3):
    files = {}  # 创建一个空字典以存储文件映射
    for f in bundle['files']: files[f['file_id']] = f['file_name']  # 将文件 ID 映射到文件名
    # 下载包中的每个文件
    for f in files:
        dl_url = f'{api}bundle/{task_id}/{f}'  # 为每个文件构建下载 URL
        retries = 0
        success = False
        while retries < max_retries and not success:  # 尝试下载，并带有重试逻辑
            try:
                dl = r.get(dl_url, headers=head, stream=True, allow_redirects=True)  # 流式下载
                if dl.status_code == 200:  # 检查请求是否成功
                    filename = files[f].split('/')[-1] if files[f].endswith('.tif') else files[f]
                    filepath = os.path.join(station_year_folder, filename)  # 定义保存文件的路径
                    if os.path.isfile(filepath):  # 如果文件已存在，则跳过下载
                        print(f"文件 {filepath} 已经存在。跳过下载。")
                        break
                    with open(filepath, 'wb') as file:  # 以二进制写入模式打开文件
                        for data in dl.iter_content(chunk_size=8192):  # 分块下载
                            file.write(data)
                    print(f"已将 {filename} 下载到 {station_year_folder}")  # 确认下载
                    success = True
                else:
                    print(f"无法下载 {filename}。状态代码：{dl.status_code}")  # 下载失败时打印错误信息
                    retries += 1
                    print(f"正在重试 {filename}... ({retries}/{max_retries})")
                    time.sleep(5)  # 等待后重试
            except Exception as e:  # 处理下载中的异常
                retries += 1
                print(f"下载 {f} 时出错：{e}。正在重试... ({retries}/{max_retries})")
                time.sleep(5)  # 等待后重试
    if not success:
        print(f"重试 {max_retries} 次后无法下载 {f}。")  # 在最大重试次数后打印失败信息
    print(f'所有可用文件已下载到：{station_year_folder}')  # 确认所有文件已下载

# 主脚本处理每个站点并下载每年的数据
for i in range(51, 100):  # 根据需要调整此范围
    station_id = nps['Name'][i]  # 获取站点名称
    nps_gc = nps[nps['Name'] == station_id].to_json()  # 将选定的站点数据转换为 GeoJSON 格式
    nps_gc = json.loads(nps_gc)  # 将 GeoJSON 解析为字典

    # 如果目录不存在，则为该站点创建输出目录
    station_destDir = os.path.join(output_dir, station_id)
    os.makedirs(station_destDir, exist_ok=True)  # 创建该目录

    years = range(nps['AmeriFlux BASE Start'][i], nps['AmeriFlux BASE End'][i])  # 确定数据提取的年份范围
    for year in years:
        if 2012 < year < 2025:  # 过滤符合条件的年份
            start_date = f"01-01-{year}"  # 定义提取数据的开始日期
            end_date = f"01-02-{year}"  # 定义提取数据的结束日期

            # 为当前年份的数据创建目录
            station_year_folder = os.path.join(station_destDir, str(year))
            os.makedirs(station_year_folder, exist_ok=True)  # 创建年份文件夹
            print(f"Folder ready: {station_year_folder}")

            # 使用开始和结束日期进行数据提取
            task = get_parameter(station_id, prodLayer, projs, nps_gc, start_date, end_date)
            task_id, bundle = get_request_polygon_shape_task1(api, task, head, station_id)  # 提交任务并获取结果
            success = download_polygon_shape_task1(api, task_id, bundle, head, station_year_folder)  # 下载文件

            if success:
                print(f"已下载 {station_id} 在年份 {year} 的数据")  # 确认下载成功
            else:
                print(f"下载数据失败 {year} 的 {station_id}。继续。")  # 处理下载失败

    print(f"完成 {station_id} 的下载。")  # 确认该站点的下载完成
