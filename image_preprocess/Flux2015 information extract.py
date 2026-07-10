
import pandas as pd
import io

import chardet


# 读取三个 CSV 文件
try:
    df1 = pd.read_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Ameriflux_site_list.csv")  # 替换为你的第一个 CSV 文件路径
    df2 = pd.read_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Our_Sites_distribution.csv")  # 替换为你的第二个 CSV 文件路径
    df3 = pd.read_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Flux2015_America.csv")  # 替换为你的第三个 CSV 文件路径
except FileNotFoundError as e:
    print(f"错误：文件未找到 - {e.filename}")
    exit()

#***************************************************
# =========================
# Step 2: 标准化列名
# =========================
if 'Site_ID' in df1.columns:
    df1.rename(columns={'Site_ID': 'SITE_ID'}, inplace=True)

# 添加缺失的时间列
for df in [df1, df2, df3]:
    for col in ['Start Year', 'End Year']:
        if col not in df.columns:
            df[col] = None

# =========================
# Step 3: 站点集合
# =========================
sites1 = set(df1['SITE_ID'])
sites2 = set(df2['SITE_ID'])
sites3 = set(df3['SITE_ID'])

# =========================
# Step 4: 分类计算
# =========================
unique_sites = {
    'Our': sites1 - sites2 - sites3,
    'Ameriflux': sites2 - sites1 - sites3,
    'Flux2015': sites3 - sites1 - sites2
}

intersect_sites = {
    'Our_Ameriflux': sites1 & sites2 - sites3,
    'Our_Flux2015': sites1 & sites3 - sites2,
    'Ameriflux_Flux2015': sites2 & sites3 - sites1
}

common_sites = sites1 & sites2 & sites3

# =========================
# Step 5: 提取站点信息函数
# =========================
# 定义函数来提取站点信息
def extract_site_info(df, site_ids):
    site_info = []
    for site_id in site_ids:
        site_data = df[df['SITE_ID'] == site_id]
        if not site_data.empty:
            # 获取所有列的数据
            site_data_dict = site_data.iloc[0].to_dict()
            site_info.append(site_data_dict)
    return pd.DataFrame(site_info)


# =========================
# Step 6: 提取并保存结果
# =========================
# 定义输出路径
output_path = "E:/HLS carbon flux/data/Sites data/Flux sites/"

# 独有站点
unique_info1 = extract_site_info(df1, unique_sites['Our'])
unique_info1 = unique_info1.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
unique_info1.to_csv(output_path + "Our_unique_sites.csv", index=False, encoding='utf_8_sig')

unique_info2 = extract_site_info(df2, unique_sites['Ameriflux'])
unique_info2 = unique_info2.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
unique_info2.to_csv(output_path + "Ameriflux_unique_sites.csv", index=False, encoding='utf_8_sig')

unique_info3 = extract_site_info(df3, unique_sites['Flux2015'])
unique_info3 = unique_info3.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
unique_info3.to_csv(output_path + "Flux2015_unique_sites.csv", index=False, encoding='utf_8_sig')

# 两两重合
intersect_info_1_2 = extract_site_info(df1, intersect_sites['Our_Ameriflux'])
intersect_info_1_2 = intersect_info_1_2.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
intersect_info_1_2.to_csv(output_path + "Our_Ameriflux_intersect.csv", index=False, encoding='utf_8_sig')

intersect_info_1_3 = extract_site_info(df1, intersect_sites['Our_Flux2015'])
intersect_info_1_3 = intersect_info_1_3.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
intersect_info_1_3.to_csv(output_path + "Our_Flux2015_intersect.csv", index=False, encoding='utf_8_sig')

intersect_info_2_3 = extract_site_info(df2, intersect_sites['Ameriflux_Flux2015'])
intersect_info_2_3 = intersect_info_2_3.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
intersect_info_2_3.to_csv(output_path + "Ameriflux_Flux2015_intersect.csv", index=False, encoding='utf_8_sig')

# 三者共有
common_info1 = extract_site_info(df1, common_sites)
common_info2 = extract_site_info(df2, common_sites)
common_info3 = extract_site_info(df3, common_sites)
common_all = pd.concat([common_info1, common_info2, common_info3], ignore_index=True)
common_all = common_all.drop_duplicates(subset=['SITE_ID'], keep='first')  # 删除重复站点
common_all.to_csv(output_path + "Common_sites_all.csv", index=False, encoding='utf_8_sig')

#
# 确保每个 DataFrame 都有公共字段，没有则添加并填充 NaN
common_fields = ['SITE_ID', 'Start Year', 'End Year', 'Latitude', 'Longitude']
for df in [df1, df2, df3]:
    for field in common_fields:
        if field not in df.columns:
            df[field] = None  # 添加缺失的列并用 NaN 填充

# 重命名 df1 中的经纬度列以匹配其他 DataFrame
if 'Lat' in df1.columns and 'Latitude' not in df1.columns:
    df1 = df1.rename(columns={'Lat': 'Latitude'})
if 'Long' in df1.columns and 'Longitude' not in df1.columns:
    df1 = df1.rename(columns={'Long': 'Longitude'})

# 提取公共字段
df1_subset = df1[common_fields]
df2_subset = df2[common_fields]
df3_subset = df3[common_fields]

# 合并三个 DataFrame
merged_df = pd.concat([df1_subset, df2_subset, df3_subset], ignore_index=True)

# 删除完全重复的行
deduplicated_df = merged_df.drop_duplicates()

# 保存结果到 CSV 文件
deduplicated_df.to_csv("E:/HLS carbon flux/data/Sites data/Flux sites/AFO_total.csv", index=False, encoding='utf_8_sig')

print("合并后的数据集已保存到 AFO_total.csv 文件")
#*****************************************************
#
# # 确保每个 DataFrame 都有 'SITE_ID' 列
# if 'Site_ID' in df1.columns:
#     df1 = df1.rename(columns={'Site_ID': 'SITE_ID'})
#
# # 确保每个 DataFrame 都有 'Start Year' 和 'End Year' 列
# # 如果不存在，则添加并填充 NaN
# for df in [df1, df2, df3]:
#     if 'Start Year' not in df.columns:
#         df['Start Year'] = None
#     if 'End Year' not in df.columns:
#         df['End Year'] = None
#
# # 获取每个 DataFrame 的站点 ID 集合
# sites1 = set(df1['SITE_ID'])
# sites2 = set(df2['SITE_ID'])
# sites3 = set(df3['SITE_ID'])
#
# # 计算独有站点
# unique_sites1 = sites1 - sites2 - sites3
# unique_sites2 = sites2 - sites1 - sites3
# unique_sites3 = sites3 - sites1 - sites2
#
# # 计算共有站点
# common_sites = sites1.intersection(sites2, sites3)
#
# # 定义函数来提取站点信息
# def extract_site_info(df, site_ids, latitude_col, longitude_col):
#     site_info = []
#     for site_id in site_ids:
#         site_data = df[df['SITE_ID'] == site_id]
#         if not site_data.empty:
#             latitude = site_data[latitude_col].iloc[0]
#             longitude = site_data[longitude_col].iloc[0]
#             start_year = site_data['Start Year'].iloc[0]
#             end_year = site_data['End Year'].iloc[0]
#             site_info.append({'SITE_ID': site_id, 'Latitude': latitude, 'Longitude': longitude,
#                               'Start Year': start_year, 'End Year': end_year})
#     return pd.DataFrame(site_info)
#
# # 提取独有站点的信息
# unique_info1 = extract_site_info(df1, unique_sites1, 'Latitude', 'Longitude')  # 替换为你实际的经纬度列名
# unique_info2 = extract_site_info(df2, unique_sites2, 'Latitude', 'Longitude')
# unique_info3 = extract_site_info(df3, unique_sites3, 'Latitude', 'Longitude')
#
# # 提取共有站点的信息
# common_info1 = extract_site_info(df1, common_sites, 'Latitude', 'Longitude')  # 替换为你实际的经纬度列名
# common_info2 = extract_site_info(df2, common_sites, 'Latitude', 'Longitude')
# common_info3 = extract_site_info(df3, common_sites, 'Latitude', 'Longitude')
#
# # 将共有站点信息合并成一个 DataFrame
# common_info = pd.concat([common_info1, common_info2, common_info3], ignore_index=True)
#
# # 保存结果到 CSV 文件
# unique_info1.to_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Our_unique_sites_2.csv", index=False, encoding='utf_8_sig')
# unique_info2.to_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Ameriflux_unique_sites_2.csv", index=False, encoding='utf_8_sig')
# unique_info3.to_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Flux2015_unique_sites_2.csv", index=False, encoding='utf_8_sig')
# common_info.to_csv("E:/HLS carbon flux/data/Sites data/Flux sites/Common_sites_2.csv", index=False, encoding='utf_8_sig')
# #
# print("结果已保存到 CSV 文件")

#*****************************************************
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         result = chardet.detect(f.read())
#     return result['encoding']
#
# file_path = "E:/HLS carbon flux/data/Sites data/Flux sites/Flux2015.csv"
# encoding = detect_encoding(file_path)
# print(f"检测到的编码为: {encoding}")
#
# import pandas as pd
# df1 = pd.read_csv(file_path, encoding=encoding)
#
# df2 = pd.read_csv("E:/HLS carbon flux/data/Sites data/Flux sites/site_start_end_years.csv")
#
# # 使用 SITE_ID 合并两个 DataFrame，使用左连接（left join），以 csv1 为基准
# merged_csv = pd.merge(df1, df2, on='SITE_ID', how='left')
#
# # 将'Start Year', 'End Year'为空的站点删除
# merged_csv = merged_csv.dropna(subset=['Start Year', 'End Year'])
# # 将结果保存为 CSV 文件
# merged_csv.to_csv('E:/HLS carbon flux/data/Sites data/Flux sites/2015_merged_data_delete_year.csv', index=False, encoding='utf_8_sig')
#
# print("合并后的数据已保存到 2015_merged_data_delete_year.csv 文件")


# # 使用 SITE_ID 合并两个 DataFrame，使用左连接（left join），以 csv1 为基准
# merged_csv = pd.merge(csv1, csv2, on='SITE_ID', how='left')
#
# # 将结果保存为 CSV 文件
# merged_csv.to_csv('merged_data.csv', index=False, encoding='utf_8_sig')
#
# print("合并后的数据已保存到 merged_data.csv 文件")
# print(df.head(5))
#
# # 将 'Year/Site ID' 列设置为索引
# df = df.set_index('Site')
#
# # 获取年份列
# years = df.columns.astype(int)
#
# # 创建一个空的 DataFrame 用于存储结果
# result = pd.DataFrame(index=df.index, columns=['Start Year', 'End Year'])
#
# # 循环遍历每一行（站点）
# for site in df.index:
#     # 找到该站点所有值为 1 的年份
#     valid_years = years[df.loc[site] == 1]
#
#     # 如果该站点有数据
#     if not valid_years.empty:
#         # 起始年份为第一个 1 对应的年份
#         result.loc[site, 'Start Year'] = valid_years.min()
#         # 结束年份为最后一个 1 对应的年份
#         result.loc[site, 'End Year'] = valid_years.max()
#     else:
#         # 如果该站点没有数据，则起始年份和结束年份都为 NaN
#         result.loc[site, 'Start Year'] = None
#         result.loc[site, 'End Year'] = None
#
# # 将结果保存为 CSV 文件
# result.to_csv('E:/HLS carbon flux/data/Sites data/Flux sites/site_start_end_years.csv', encoding='utf_8_sig')
#
# print("结果已保存到 site_start_end_years.csv 文件")