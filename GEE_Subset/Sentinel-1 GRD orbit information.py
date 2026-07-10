import pandas as pd
import ee
from datetime import datetime, timedelta
import os
import time

# # 初始化 GEE
# service_account = 'harvey1989lillian-gmail-com@biomass-estimates.iam.gserviceaccount.com'
# credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/个人重要文件/GEE_APIKER/biomass-estimation/biomass-estimates-d58a1f0e77e5.json")
# ee.Initialize(credentials)
service_account = 'gpp-estimation-dl@gpp-estimation.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "E:/Personal infomation/个人重要文件/GEE_APIKER/GPP_estimation_DL/gpp-estimation-a13404c05e60.json")
ee.Initialize(credentials)
#*****************************************************************************
# Sentinel-1 集合
S1_COLLECTION = "COPERNICUS/S1_GRD"

# 读取站点数据
site_data = pd.read_csv("E:/Sentinel SAR/id_csv/11SKA_sample_locs.csv")

# 定义时间范围
start_date = datetime(2014, 1, 1)  # Sentinel-1 数据有效起始日期
end_date = datetime(2023, 12, 31)

# 确保输出目录存在
output_dir = "E:/Sentinel SAR/orbit_angle/11SKA_sample_locs"
os.makedirs(output_dir, exist_ok=True)

# 遍历站点
for i in range(102,150):  # 使用 range(len(site_data)) 遍历所有行
    row = site_data.iloc[i]  # 获取当前行
    site_id = row['id']
    lat, lon = row['lat'], row['lon']
    point = ee.Geometry.Point([lon, lat])

    # 遍历每个年份
    current_year = start_date.year
    while current_year <= end_date.year:
        # 定义每年的开始和结束日期
        current_date = datetime(current_year, 1, 1)
        next_date = datetime(current_year + 1, 1, 1)

        # 生成要保存的文件名
        output_file = os.path.join(output_dir, f"{site_id}_{current_year}.csv")

        # 如果文件已存在，跳过当前年份的数据检索
        if os.path.exists(output_file):
            print(f"文件已存在，跳过处理: {output_file}")
            current_year += 1
            continue

        # 存储当前年份的数据
        results = []

        # 遍历当前年份中的每一天
        while current_date < next_date:
            # 检索 Sentinel-1 数据
            max_attempts = 5  # 最大重试次数
            attempt = 0
            while attempt < max_attempts:
                try:
                    s1_data = ee.ImageCollection(S1_COLLECTION) \
                        .filterBounds(point) \
                        .filterDate(current_date.strftime("%Y-%m-%d"), (current_date + timedelta(days=1)).strftime("%Y-%m-%d"))

                    # 检查数据集是否为空
                    data_size = s1_data.size().getInfo()
                    break  # 如果成功获取数据大小，就退出重试循环
                except Exception as e:
                    attempt += 1
                    print(f"请求数据失败，尝试第 {attempt} 次。错误信息: {e}")
                    time.sleep(5)  # 等待5秒后重试

            if data_size > 0:
                # 获取数据并提取信息
                s1_list = s1_data.toList(data_size).getInfo()
                for img in s1_list:
                    orbit_direction = img['properties']['orbitProperties_pass']
                    product_name = img['id']  # 获取产品名称（system:index）

                    # 加载图像并提取 `angle` 波段值
                    img_id = img['id']
                    image = ee.Image(img_id)
                    angle_value = image.select('angle').reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=10  # 提取的分辨率，可根据需要调整
                    ).get('angle').getInfo()

                    results.append({
                        'id': site_id,
                        'lat': lat,
                        'lon': lon,
                        'date': current_date.strftime("%Y-%m-%d"),
                        'product_name': product_name,
                        'orbit_direction': orbit_direction,
                        'angle_value': angle_value if angle_value is not None else "NO_DATA",
                    })
            else:
                # 如果没有数据，记录为 NO_DATA
                results.append({
                    'id': site_id,
                    'lat': lat,
                    'lon': lon,
                    'date': current_date.strftime("%Y-%m-%d"),
                    'product_name': "NO_DATA",
                    'orbit_direction': "NO_DATA",
                    'angle_value': "NO_DATA",
                })

            current_date += timedelta(days=1)

        # 将结果保存到 CSV 文件
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

        # 打印每年完成消息
        print(f"第 {i + 1} 个站点第 {current_year} 年处理完成: {site_id}")

        # 处理下一个年份
        current_year += 1

print("所有站点的数据已成功保存到指定目录。")

#********************************************
# 获取该站点 该时间所对应的轨道信息（上升或者下降）+角度信息 所有数据存于一个文件
# Sentinel-1 数据集
# S1_COLLECTION = "COPERNICUS/S1_GRD"
#
# # 读取站点数据
# site_data = pd.read_csv("E:/Sentinel SAR/id_csv/11SKA_sample_locs.csv")
#
# # 定义时间范围
# start_date = datetime(2015, 10, 6)  # Sentinel-1 数据有效起始日期
# end_date = datetime(2015, 10, 15)
#
# # 存储结果
# results = []
#
# # 遍历站点
# # for index, row in site_data.iterrows():
# for index, row in site_data.iloc[:2].iterrows():  # 取前 3 行
#     site_id = row['id']
#     lat, lon = row['lat'], row['lon']
#     point = ee.Geometry.Point([lon, lat])
#
#     # 遍历时间范围
#     current_date = start_date
#     while current_date <= end_date:
#         next_date = current_date + timedelta(days=1)
#
#         # 检索 Sentinel-1 数据
#         s1_data = ee.ImageCollection(S1_COLLECTION) \
#             .filterBounds(point) \
#             .filterDate(current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d"))
#
#         # 检查数据集是否为空
#         data_size = s1_data.size().getInfo()
#         if data_size > 0:
#             # 获取数据并提取信息
#             s1_list = s1_data.toList(data_size).getInfo()
#             for img in s1_list:
#                 orbit_direction = img['properties']['orbitProperties_pass']
#                 product_name = img['id']  # 获取产品名称（system:index）
#
#                 # 加载图像并提取 `angle` 波段值
#                 img_id = img['id']
#                 image = ee.Image(img_id)
#                 angle_value = image.select('angle').reduceRegion(
#                     reducer=ee.Reducer.mean(),
#                     geometry=point,
#                     scale=10  # 提取的分辨率，可根据需要调整
#                 ).get('angle').getInfo()
#
#                 results.append({
#                     'id': site_id,
#                     'lat': lat,
#                     'lon': lon,
#                     'date': current_date.strftime("%Y-%m-%d"),
#                     'product_name': product_name,
#                     'orbit_direction': orbit_direction,
#                     'angle_value': angle_value if angle_value is not None else "NO_DATA",
#                 })
#         else:
#             # 如果没有数据，记录为 NO_DATA
#             results.append({
#                 'id': site_id,
#                 'lat': lat,
#                 'lon': lon,
#                 'date': current_date.strftime("%Y-%m-%d"),
#                 'product_name': "NO_DATA",
#                 'orbit_direction': "NO_DATA",
#                 'angle_value': "NO_DATA",
#             })
#
#         current_date = next_date
#
# # 保存结果到 CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("E:/Sentinel SAR/id_csv/11SKA_orbit_directions_with_angle.csv", index=False)

#********************************************
# 获取该站点 该时间所对应的轨道信息（上升或者下降）
# # Sentinel-1 数据集
# S1_COLLECTION = "COPERNICUS/S1_GRD"
#
# # 读取站点数据
# site_data = pd.read_csv("E:/Sentinel SAR/id_csv/11SKA_sample_locs.csv")
#
# # 定义时间范围
# start_date = datetime(2015, 10, 6)  # Sentinel-1 数据有效起始日期
# end_date = datetime(2015, 10, 10)
#
# # 存储结果
# results = []
#
# # 遍历站点
# # for index, row in site_data.iterrows():
# for index, row in site_data.iloc[:2].iterrows():  # 取前 3 行
#     site_id = row['id']
#     lat, lon = row['lat'], row['lon']
#     point = ee.Geometry.Point([lon, lat])
#
#     # 遍历时间范围
#     current_date = start_date
#     while current_date <= end_date:
#         next_date = current_date + timedelta(days=1)
#
#         # 检索 Sentinel-1 数据
#         s1_data = ee.ImageCollection(S1_COLLECTION) \
#             .filterBounds(point) \
#             .filterDate(current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d"))
#
#         # 检查数据集是否为空
#         data_size = s1_data.size().getInfo()
#         if data_size > 0:
#             # 获取数据并提取信息
#             s1_list = s1_data.toList(data_size).getInfo()
#             for img in s1_list:
#                 orbit_direction = img['properties']['orbitProperties_pass']
#                 product_name = img['id']  # 获取产品名称（system:index）
#                 results.append({
#                     'id': site_id,
#                     'lat': lat,
#                     'lon': lon,
#                     'date': current_date.strftime("%Y-%m-%d"),
#                     'product_name': product_name,
#                     'orbit_direction': orbit_direction,
#                 })
#         else:
#             # 如果没有数据，记录为 NO_DATA
#             results.append({
#                 'id': site_id,
#                 'lat': lat,
#                 'lon': lon,
#                 'date': current_date.strftime("%Y-%m-%d"),
#                 'product_name': "NO_DATA",
#                 'orbit_direction': "NO_DATA",
#             })
#
#         current_date = next_date
#
# # 保存结果到 CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("E:/Sentinel SAR/id_csv/sentinel1_orbit_directions.csv", index=False)
