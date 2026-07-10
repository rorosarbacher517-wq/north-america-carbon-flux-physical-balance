import pandas as pd
import rasterio as rio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.shutil import copy
import requests as r
from pyproj import Proj
import numpy as np

# data retrieve
# RS data
# # 将同一个站点的时间序列数据合并到一起
L30_df = pd.read_csv("E:/HLS carbon flux/data/HLS_Time_series/HLS-253-263-HLSL30-020-results.csv")
S30_df = pd.read_csv("E:/HLS carbon flux/data/HLS_Time_series/HLS-253-263-HLSS30-020-results.csv")
#
# # 提取各自所需的波段
# # L30选择Category	ID	Latitude	Longitude	Date  HLSL30_020_B01	HLSL30_020_B02	HLSL30_020_B03	HLSL30_020_B04	HLSL30_020_B05	HLSL30_020_B06	HLSL30_020_B07 HLSL30_020_Fmask
# # S30 选择Category	ID	Latitude	Longitude	Date HLSS30_020_B01	HLSS30_020_B02	HLSS30_020_B03	HLSS30_020_B04	HLSS30_020_B8A	HLSS30_020_B11	HLSS30_020_B12	HLSS30_020_Fmask
# #
# # 将L30的HLSL30_020_B01	HLSL30_020_B02	HLSL30_020_B03	HLSL30_020_B04	HLSL30_020_B05	HLSL30_020_B06	HLSL30_020_B07 HLSL30_020_Fmask重命名为 coastal aerosal;blue;green; red; nir; swir1;swir2;
# # 将S30 的HLSS30_020_B01	HLSS30_020_B02	HLSS30_020_B03	HLSS30_020_B04	HLSS30_020_B8A	HLSS30_020_B11	HLSS30_020_B12	HLSS30_020_Fmask重命名为 coastal aerosal;blue;green; red; nir; swir1;swir2;
# #
# #
# # 将两个重命名后的df拼接在一起，
# #
# # 同一个站点站点ID	Latitude	Longitude的按照Date进行排序
# #
# # 打印排序后的df的前10行
#
# 提取并重命名L30的波段
L30_df = L30_df[['Category','ID', 'Latitude', 'Longitude', 'Date',
                  'HLSL30_020_B01', 'HLSL30_020_B02', 'HLSL30_020_B03',
                  'HLSL30_020_B04', 'HLSL30_020_B05', 'HLSL30_020_B06',
                  'HLSL30_020_B07', 'HLSL30_020_Fmask']]

L30_df.columns = ['Category','ID', 'Latitude', 'Longitude', 'Date',
                   'coastal aerosal', 'blue', 'green', 'red',
                   'nir', 'swir1', 'swir2', 'Fmask']

# 提取并重命名S30的波段
S30_df = S30_df[['Category','ID', 'Latitude', 'Longitude', 'Date',
                  'HLSS30_020_B01', 'HLSS30_020_B02', 'HLSS30_020_B03',
                  'HLSS30_020_B04', 'HLSS30_020_B8A', 'HLSS30_020_B11',
                  'HLSS30_020_B12', 'HLSS30_020_Fmask']]

S30_df.columns = ['Category','ID', 'Latitude', 'Longitude', 'Date',
                   'coastal aerosal', 'blue', 'green', 'red',
                   'nir', 'swir1', 'swir2', 'Fmask']

# 将两个DataFrame拼接在一起
combined_df = pd.concat([L30_df, S30_df], ignore_index=True)

# 按照站点ID、纬度、经度和日期进行排序
combined_df.sort_values(by=['Category','ID', 'Latitude', 'Longitude', 'Date'], inplace=True)

# # combined_df在后面加一列，这列的值是根据Fmask标识的该像素的质量。将Fmask列的值转为二进制，如果二进制的bit1,2,3,4,5,6,7中任何一个含有1，则标记为1（bad-quality）；否则则标记为0（good-quality）
# # 确保 Fmask 列为整数类型，并处理缺失值
combined_df['Fmask'] = combined_df['Fmask'].fillna(0).astype(int)

#
def quality_indicator(fmask_value):
    # 转换为二进制字符串，去掉 '0b' 前缀，左侧用0填充到8位
    binary_str = format(fmask_value, '08b')

    # 检查特定的位 (bit索引从0开始，因此对应的索引与描述稍有不同)
    if binary_str[1] == '1' or binary_str[3] == '1' or binary_str[4] == '1' or binary_str[5] == '1':
        return 1, binary_str  # bad quality
    else:
        return 0, binary_str  # good quality
#
# def extract_quality_and_bits(fmask_value):
#     bitword_order = (1, 1, 1, 1, 1, 1, 2)  # 每个bitword的位数
#     num_bitwords = len(bitword_order)  # bitwords数量
#     total_bits = sum(bitword_order)  # 总位数，应该是8, 16, 或 32
#
#     # 转换为二进制字符串并左侧填充零
#     bit_val = format(fmask_value, 'b').zfill(total_bits)
#
#     all_bits = []
#     good_quality = 0
#
#     # 分离每个位字的值
#     bits = total_bits
#     for i, b in enumerate(bitword_order):
#         prev_bit = bits
#         bits -= b
#
#         if i == 0:  # 第一个bitword
#             bitword = bit_val[bits:]  # 提取最高位
#         elif i == num_bitwords - 1:  # 最后一个bitword
#             bitword = bit_val[:prev_bit]  # 提取最低位
#         else:
#             bitword = bit_val[bits:prev_bit]  # 提取中间位
#
#         all_bits.append(int(bitword, 2))  # 将提取的位转为整数
#
#         # 输出bitword（可选，仅用于调试）
#         print(f' Bit Word {i + 1}: {bitword}')
#
#     # 检查特定的位位于2, 4, 5, 6
#     if all_bits[2] + all_bits[4] + all_bits[5] + all_bits[6] == 0:
#         good_quality = 1  # 表示好质量
#
#     return good_quality, bit_val  # 返回质量和二进制字符串


# 示例: 应用函数并增加新列
# combined_df[['Quality', 'Binary_Fmask']] = combined_df['Fmask'].apply(extract_quality_and_bits).apply(pd.Series)

# # 应用函数并增加新列
combined_df[['Quality', 'Binary_Fmask']] = combined_df['Fmask'].apply(quality_indicator).apply(pd.Series)# # 使用 apply 函数将判断应用到 Fmask 列
# combined_df['Quality'] = combined_df['Fmask'].apply(quality_indicator)

# # 打印添加 Quality 列后的 DataFrame 的前 10 行
print(combined_df.head(10))

# Go through & split out the values for each bit word based on input above:

# bitword_order = (1, 1, 1, 1, 1, 1, 2)
# total_bits = sum(bitword_order)
# bit_val = str(10000000)
# all_bits = []
# num_bitwords = len(bitword_order)
# bits = total_bits
# i = 0
# for b in bitword_order:
#     prev_bit = bits
#     bits = bits - b
#     i = i + 1
#     if i == 1:
#         bitword = bit_val[bits:]
#         print(' Bit Word ' + str(i) + ': ' + str(bitword))
#         all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
#     elif i == num_bitwords:
#         bitword = bit_val[:prev_bit]
#         print(' Bit Word ' + str(i) + ': ' + str(bitword))
#         all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
#     else:
#         bitword = bit_val[bits:prev_bit]
#         print(' Bit Word ' + str(i) + ': ' + str(bitword))
#         all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))

 # flux data
# inputdata_path ='./data/sites_input/'
# y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
# y_qa_array_all = np.load(inputdata_path + 'All_sites_y_qa.npy', allow_pickle=True)
#
# # Search for the HLSS30 items of interest:
# search_query4 = f"{search_query3}&collections={s30_id}"
# s30_items = r.get(search_query4).json()['features']
# # Search for the HLSL30 items of interest:
# search_query5 = f"{search_query3}&collections={l30_id}"
# l30_items = r.get(search_query5).json()['features']
# hls_items = s30_items + l30_items  # Combine the S30 ad L30 items:
#
# h = hls_items[0]
# evi_band_links = []
#
# # Define which HLS product is being accessed
# if h['assets']['browse']['href'].split('/')[4] == 'HLSS30.015':
#     evi_bands = ['B01', 'B02', 'B03','B04', 'B8A', 'B11','B12', 'Fmask'] # NIR RED BLUE Quality for S30
# else:
#     evi_bands = ['B01', 'B02', 'B03','B04', 'B05', 'B06','B07', 'Fmask'] # NIR RED BLUE Quality for L30
#
# # Subset the assets in the item down to only the desired bands
# for a in h['assets']:
#     if any(b == a for b in evi_bands):
#         evi_band_links.append(h['assets'][a]['href'])
# for e in evi_band_links: print(e)
#
# # Read the file using rasterio.
# # Use vsicurl to load the data directly into memory (be patient, may take a few seconds)
# for e in evi_band_links:
#     if e.rsplit('.', 2)[-2] == evi_bands[0]: # NIR index
#         nir = rio.open(e)
#     elif e.rsplit('.', 2)[-2] == evi_bands[1]: # red index
#         red = rio.open(e)
#     elif e.rsplit('.', 2)[-2] == evi_bands[2]: # blue index
#         blue = rio.open(e)
#     elif e.rsplit('.', 2)[-2] == evi_bands[3]: # Fmask index
#         fmask = rio.open(e)
# print("The COGs have been loaded into memory!")
#
# geo_CRS = Proj('+proj=longlat +datum=WGS84 +no_defs', preserve_units=True)  # Source coordinate system of the ROI
# utm = pyproj.Proj(nir.crs)                                                  # Destination coordinate system
# project = pyproj.Transformer.from_proj(geo_CRS, utm)                        # Set up the transformation
# fsUTM = transform(project.transform, fieldShape)                            # Apply reprojection
# fmask_array, _ = rio.mask.mask(fmask, [fsUTM], crop=True)  # Load in the Quality data