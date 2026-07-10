import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def convert_s_d_mean(flux):
    return flux * 86400 * 12 * 1e-6

# 半小时的转为一天的
def custom_day_mean(nees):
    valid_values = nees[nees != -9999]
    if valid_values.empty:
        hourly_avg_converted = -9999
    else:
        # 针对'NEE_uStar_f', 'GPP_uStar_f', 'Reco_uStar',三个变量
        # 进行单位转换时，先求每48个半小时的平均值 不进行单位转换时，直接对这48个值求均值就好
        hourly_avg = valid_values.mean()
        return hourly_avg

if __name__ == "__main__":
    # # ************************************
    # # # step2 将有效站点信息与y_qa进行匹配
    # sites_df = pd.read_csv('E:/Paper code/code/Carbon_flux_estimation/data/sites_y_quality_match/All_sites_info.csv', low_memory=False)
    # qa_path = 'E:/Carbon_flux/data/originate_data/GE_AmeriFlux_data/GE_AmeriFlux_data/'
    # sites_qa_doy_list = []
    # site_name_list = sites_df['Site_Id'].unique()
    # # 排除值为-9999的站点名称
    # site_name_list = site_name_list[site_name_list != '-9999']
    # for i in range(0,len(site_name_list)): # len(site_name_list)
    #     site_name = site_name_list[i]
    #     qa_file_name = 'GE_AMF_' + site_name + '_HH'
    #     qa_file_path = os.path.join(qa_path,qa_file_name +'.txt')
    #     qa_df = pd.read_csv(qa_file_path, delimiter="\t", low_memory=False)
    #     # print(qa_df)
    #     # 删除第一行 单位行
    #     qa_df.drop(index=0, inplace=True)
    #     # print(qa_df.head())
    #     # 将 "qa_df" 数据帧中的所有列转换为数值类型
    #     qa_df = qa_df.apply(pd.to_numeric, errors='coerce')
    #     # 将-9999替换为空值
    #     qa_df.replace(-9.9990e+03, np.nan, inplace=True)
    #     # print(qa_df.head())
    #     # 重新设置索引
    #     qa_df.reset_index(inplace=True)
    #     # 根据year和doy进行分组，然后对NEE_uStar_fqc ，GPP_uStar_fqc 求均值
    #     qa_df_DD = qa_df.groupby(["Year", "DoY"]).agg({"NEE_uStar_fqc": custom_day_mean,
    #                                                     "GPP_uStar_fqc": custom_day_mean
    #                                                     }).rename(
    #         columns={"NEE_uStar_fqc": "NEE_uStar_fqc_d_mean",
    #                  "GPP_uStar_fqc": "GPP_uStar_fqc_d_mean",
    #                  })
    #     # 还原 "Year" 和 "DoY" 作为列
    #     qa_df_DD.reset_index(inplace=True)
    #     qa_df_DD['Site_Id'] = site_name
    #     # sites_qa_doy = sites_df.merge(qa_df_DD, on=['Site_Id',"Year", "DoY"])
    #     print(qa_df_DD.head())
    #     # 将sites_qa_doy添加到list
    #     sites_qa_doy_list.append(qa_df_DD)
    # # print(sites_qa_doy_list)
    # # 使用 concat 方法合并所有部分
    # merged_sites_qa_doy = pd.concat(sites_qa_doy_list, ignore_index=True)
    # # 输出结果
    # # 将merged_sites_qa_doy与sites_df进行匹配
    # # 使用 pandas 的 merge 方法，指定 how='left'，这样将只保留 'sites_df' 中原来为 -9999 的部分
    # merged_sites_qa = sites_df.merge(merged_sites_qa_doy, how='left', on=['Site_Id', 'Year', 'DoY'])
    # # 将 'sites_df' 中原来为 -9999 的部分更新回 -9999
    # merged_sites_qa.fillna(-9999, inplace=True)
    # merged_sites_qa.to_csv('E:/Paper code/code/Carbon_flux_estimation/data/sites_y_quality_match/All_sites_y_qa.csv', index=False)
    merged_sites_qa = pd.read_csv('E:/Paper code/code/Carbon_flux_estimation/data/sites_y_quality_match/All_sites_y_qa.csv')
    merged_sites_qa_array = np.array(merged_sites_qa)
    print(merged_sites_qa_array.shape)
    # 将其转为site-year,doy，feature多维数组
    reshaped_array = np.reshape(merged_sites_qa_array, (1769, 366, 5))
    print(reshaped_array.shape)
    reshaped_array[:,:,3]
    np.save('E:/Paper code/code/Carbon_flux_estimation/data/sites_y_quality_match/All_sites_y_qa.npy',reshaped_array)
    # ************************************
    # step1 将站点名-年份-doy保存为csv
    # download the data
    # inputdata_path = 'D:/Carbon_flux/data/sites_images_input/'
    # # linear dl
    # y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
    # print(y_image_array_all.shape)
    # # # 获取每个站点的信息 用于匹配flag
    # sites_info = y_image_array_all[:,:,[0,3,4]]
    # # 将信息存为df.csv
    # # 创建一个 DataFrame 包含三个字段
    # df = pd.DataFrame({
    #     'Site_Id': sites_info[:, :, 0].flatten(),
    #     'Year': sites_info[:, :, 1].flatten(),
    #     'DoY': sites_info[:, :, 2].flatten()
    # })
    # # 将 DataFrame 保存为 CSV 文件
    # df.to_csv('E:/Paper code/code/Carbon_flux_estimation/data/sites_y_quality_match/All_sites_info.csv', index=False)

