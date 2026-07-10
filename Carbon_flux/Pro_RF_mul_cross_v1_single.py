# 随机森林同时预测5倍交叉验证

import pandas as pd
import os
import Data_preprocess
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from keras import backend as K
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
import Data_filter_match
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def y_statistic_df(y_image_array):
    # 提取除为字符串的第0 第16,17列以外的所有列
    data = y_image_array[:, :, np.r_[1:16, 18:23]]
    data = data.astype(float)  # 将数据类型转换为浮点数
    data = data.reshape(-1, 20)  # 将数据展平为二维数组，每行20个特征
    # 忽略等于-9999的填充值
    data[data == -9999.0] = np.nan
    # 计算均值、标准差、最大值和最小值（忽略NaN值）
    mean_values = np.nanmean(data, axis=0)
    std_values = np.nanstd(data, axis=0)
    max_values = np.nanmax(data, axis=0)
    min_values = np.nanmin(data, axis=0)
    # 创建 DataFrame 对象
    df_data = pd.DataFrame({
        'mean': mean_values,
        'std': std_values,
        'max': max_values,
        'min': min_values
    })
    # 输出 DataFrame
    statistic_dir = './data/sites_statistic/'
    if not os.path.isdir(statistic_dir):
        # 创建文件夹
        os.makedirs(statistic_dir)
        print(f"已创建文件夹：{statistic_dir}")
    else:
        print(f"文件夹已存在：{statistic_dir}")
    df_data.to_csv(statistic_dir + 'y_statistic.csv')
    # print(df_data)


def replace_veg_cli(covariate_array):
    # covariate_array[covariate_array == -9999] = -9999.0
    unique_values_veg = np.unique(covariate_array[:, :, 7][covariate_array[:, :, 7] != -9999].astype(str))
    unique_values_clim = np.unique(covariate_array[:, :, 8][covariate_array[:, :, 8] != -9999].astype(str))
    replacement_dict_veg = {value: index for index, value in enumerate(unique_values_veg)}
    replacement_dict_clim = {value: index for index, value in enumerate(unique_values_clim)}
    # 将这个字典输出保存为txt
    statistic_dir = './data/sites_statistic/'
    if not os.path.isdir(statistic_dir):
        # 创建文件夹
        os.makedirs(statistic_dir)
        print(f"已创建文件夹：{statistic_dir}")
    else:
        print(f"文件夹已存在：{statistic_dir}")
    with open(os.path.join(statistic_dir, "replacement_dict_veg.txt"), "w") as f:
        for key, value in replacement_dict_veg.items():
            f.write(f"{key}:{value}\n")
    with open(os.path.join(statistic_dir, "replacement_dict_clim.txt"), "w") as f:
        for key, value in replacement_dict_clim.items():
            f.write(f"{key}:{value}\n")
    for i in range(covariate_array.shape[0]):
        for j in range(covariate_array.shape[1]):
            if covariate_array[i, j, 7] != -9999.0:
                covariate_array[i, j, 7] = replacement_dict_veg[str(covariate_array[i, j,7])]
            if covariate_array[i, j, 8] != -9999.0:
                covariate_array[i, j, 8] = replacement_dict_clim[str(covariate_array[i, j, 8])]
    return covariate_array


def process_pixel_values(input_array):
    # output_array = np.zeros_like(input_array)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            # for m in range(input_array.shape[2]):
            window_values = input_array[i, j, :, :, :]
            if np.any(window_values != -9999):
                input_array[i, j, :, :, :] = np.where(window_values == -9999, 0, window_values)
            else:
                input_array[i, j, :, :, :] = -9999
    return input_array


## real,pred = y_test_plot_i, y_pred_plot_i
def get_regression_line(real, pred, data_range=(0, 110)):
    # 拟合（若换MK，自行操作）最小二乘
    def slope(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

    k, b = slope(real, pred)
    regression_linex = []
    regression_liney = []
    for a in range(int(data_range[0]), int(data_range[1]) + 1):
        # print (a)
        regression_liney.append(a)
        regression_linex.append((k * a) + b)
    return regression_linex, regression_liney


def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999.0).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

def get_inter_array(x_ref_array,is_inter):
    # 创建一个与x_ref_array形状相同的mask数组，并初始化为-9999
    nointer_array = np.full_like(x_ref_array[...], -9999)
    inter_array = np.full_like(x_ref_array[...], -9999)
    all_array = x_ref_array.copy()
    # 根据最后一个特征值创建mask
    for i in range(x_ref_array.shape[0]):
        for j in range(x_ref_array.shape[1]):
            for m in range(x_ref_array.shape[3]):
                for n in range(x_ref_array.shape[4]):
                    if x_ref_array[i, j, -1, m, n] == 0:
                        nointer_array[i, j, :, m, n] = np.where(nointer_array[i, j, :, m, n] == -9999,
                                                                x_ref_array[i, j, :, m, n], -9999)
                    elif x_ref_array[i, j, -1, m, n] == 1:
                        inter_array[i, j, :, m, n] = np.where(inter_array[i, j, :, m, n] == -9999,
                                                              x_ref_array[i, j, :, m, n], -9999)
                    # 对于特征值为2的情况，保持不变
                    else:
                        all_array[i, j, :, m, n] = x_ref_array[i, j, :, m, n]
    if is_inter == 0:
        x_array = nointer_array[..., :-1, :, :]
    elif is_inter == 1:
        x_array = inter_array[..., :-1, :, :]
    else:
        x_array = all_array[..., :-1, :, :]
    return x_array

def full_intervalue(interpolated_array):
    interpolated_array1 = interpolated_array.copy()
    invalid_indices = np.any((interpolated_array1[:, :, :, :, :] == -9999) | (interpolated_array1[:, :, :, :, :] == 1), axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array1.shape[2], axis=2)
    interpolated_array1[:,:,0,1,1]
    interpolated_array1[reshaped_indices] = -9999
    return interpolated_array1


# 给定影像的根路径
if __name__ == "__main__":
    # download the data
    # 插值的数据
    # download the data
    inputdata_path = './data/sites_input/'
    x_ref_array = np.load(inputdata_path + 'x_ref_modis_interpolation_mark.npy', allow_pickle=True)
    x_lai_array = np.load(inputdata_path + 'x_lai_modis_interpolation_mark.npy', allow_pickle=True)
    x_mete_array = np.load(inputdata_path + 'x_mete_era5_interpolation_mark.npy', allow_pickle=True)
    y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
    y_qa_array_all = np.load(inputdata_path + 'All_sites_y_qa.npy', allow_pickle=True)
    x_image_array_all = np.concatenate(
        (x_ref_array[..., :-1, :, :], x_lai_array[..., :-1, :, :], x_mete_array[..., :-1, :, :]), axis=2)
    x_inter_mask_all = np.concatenate(
        (x_ref_array[..., -1:, :, :], x_lai_array[..., -1:, :, :], x_mete_array[..., -1:, :, :]), axis=2)
    ##### 匹配前检查同一像素不同波段值缺失的情况
    x_inter_mask_all[:, :, 1, 1, 1]
    y_image_array_all[:, :, 0]
    x_lai_array[:, :, 0, 1, 1]
    y_qa_array_all[:, :, 3]
    # 将y_image_array_all进行特征填充
    x_image_array, y_image_array, x_inter_mask_array0, y_qa_array = Data_filter_match.mul_match_x_y_modis(
        x_image_array_all,
        y_image_array_all,
        x_inter_mask_all,
        y_qa_array_all,
        ref=True, lai=True,
        meteorology=False,
        era5_meteorology=True,
        NEE_GPP_RECO=True,
        ref_quality=0,
        ref_filledvalue=0,
        lai_filledvalue=0,
        lai_quality=0,
        era5_mete_filledvalue=0,
        meteorology_filledvalue=0,
        NEE_GPP_RECO_filledvalue=0)
    print(x_image_array.shape, y_image_array.shape)
    # x_image_array = full_filledvalue(x_image_array)
    x_inter_mask_array0[:, :, 0, 1, 1]
    y_qa_array[:, :, 3]
    x_inter_mask_array = full_intervalue(x_inter_mask_array0)
    y_flux_qa_array = y_qa_array[:, :, -5:]
    y_flux_qa_array[:, :, 0]
    x_inter_mask_array0[:, :, 0, 1, 1]
    # 将交叉验证的结果保存下来进行验证分析
    # results_variation_path = './data/sites_estimate_result/'
    results_variation_path = './data/sites_results/'
    if not os.path.isdir(results_variation_path):
        # 创建文件夹
        os.makedirs(results_variation_path)
        print(f"已创建文件夹：{results_variation_path}")
    np.save(results_variation_path + 'RF_RL_NEE_x_mask_soil.npy', x_inter_mask_array)
    np.save(results_variation_path + 'RF_RL_NEE_y_qa_soil.npy', y_flux_qa_array)
    # 将质量标识的波段去除
    # x_image_array = x_image_array[:,:,[0,1,2,3,4,5,6,14,16,17,18,19,20,21,22],:,:]
    check_dim_fill(x_image_array)
    # ******************************************
    # 对不符合条件的噪声值进行预处理
    covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12,3,5,16,17,18, 19, 20, 21,22]]
    # # 分别对应DoY，Rg_daily，PotRad_uStar_daily，Tair_f_daily，VPD_f_daily，Sensor_id，Year，Month，
    # # Vegetation_Abbreviation(IGBP)，Climate_Class_Abbreviation(Koeppen)，Mean_Average_Precipitation(mm)，
    # # Mean_Average_Temperature(degreesC)，Latitude(degrees)，Longitude(degrees)，Elevation(m)
    # # 统计y_image_array的特征
    y_statistic_df(y_image_array)
    # # 将covariate_array中的植被类型（新array中的第8列）和气候类型（新array中的第8列）换成离散数字表示，因为字符串类型计算不了均值等统计值
    covariate_array = replace_veg_cli(covariate_array)
    # # # # 根据index划分训练集和测试集
    # index_train, index_test, trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, RF_trainy, RF_testy = Data_preprocess.get_training_test_com2 \
    #     (x_image_array, y_image_array, covariate_array, proportion=0.8, split_by_site=True)
    # print(trainx.shape, trainy.shape, testx.shape, testy.shape)
    # k倍交叉验证划分数据集
    cross_validation_data = Data_preprocess.get_cross_validation_training_test(x_image_array, y_image_array,
                                                                               covariate_array)
    Y_pred_filtered = []
    Y_test_filtered = []
    RF_test_sites = []
    Rmse = []
    R2 = []
    CC = []
    Bias = []
    N = []
    # 依次遍历划分好的训练集和测试集
    for kth_data in cross_validation_data:
        # index_train, index_test, trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, RF_trainy, RF_testy = cross_validation_data
        trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, RF_trainy, RF_testy,test_sites_array = (
            kth_data['trainx'], kth_data['trainy'], kth_data['testx'], kth_data['testy'],
            kth_data['train_covariate'], kth_data['test_covariate'], kth_data['train_sites_counts'],
            kth_data['test_sites_counts'], kth_data['RF_trainy'],
            kth_data['RF_testy'],kth_data['test_array'])
        RF_testy = RF_testy[:, :, :].reshape(RF_testy.shape[0], RF_testy.shape[1], 3)
        RF_trainy = RF_trainy[:, :, :].reshape(RF_trainy.shape[0], RF_trainy.shape[1], 3)
        ## 对训练集和测试集的影像数据和其他协变量进行标准化
        input_images_train_norm0, input_images_test_norm0, input_covariate_train_norm0, input_covariate_test_norm0, mean_train, std_train, mean_train_cov, std_train_cov = Data_preprocess.construct_composite_train_test(
            trainx,
            testx, train_covariate, test_covariate,
            is_single_norm=True,
            is_train_test_com=True)
        # print(input_covariate_train_norm0,input_covariate_test_norm0)

        # 对输入影像进行均值平滑处理
        input_images_train_norm1 = process_pixel_values(input_images_train_norm0)
        RF_input_images_train_norm1 = np.mean(input_images_train_norm1.reshape(-1, 3, 3), axis=(-2, -1))
        # reshape
        RF_input_images_train_norm2 = RF_input_images_train_norm1.reshape(input_images_train_norm0.shape[0],
                                                                          input_images_train_norm0.shape[1],
                                                                          input_images_train_norm0.shape[2])
        # 若考虑气象协变量Rg_daily，PotRad_uStar_daily，Tair_f_daily，VPD_f_daily
        # RF_input_images_train_norm3 = np.concatenate(
        #     (RF_input_images_train_norm2, input_covariate_train_norm0[:, :, 1:5]), axis=2)
        # 采用了era5land气象数据之后，不用再concatenate
        RF_input_images_train_norm3 = RF_input_images_train_norm2
        # reshape
        RF_input_images_train_norm4 = RF_input_images_train_norm3.reshape(
            RF_input_images_train_norm3.shape[0] * RF_input_images_train_norm3.shape[1],
            RF_input_images_train_norm3.shape[2])
        RF_input_images_train_norm4.shape
        # 对自变量值为-9999的数据进行处理
        RF_train_valid_indices = np.where((RF_input_images_train_norm4 != -9999).all(axis=1))
        RF_input_images_train_norm4_clean = RF_input_images_train_norm4[RF_train_valid_indices]
        RF_trainy = RF_trainy.reshape(RF_trainy.shape[0] * RF_trainy.shape[1], 3)
        RF_trainy_clean = RF_trainy[RF_train_valid_indices]
        # 对输入影像进行均值平滑处理
        input_images_test_norm1 = process_pixel_values(input_images_test_norm0)
        RF_input_images_test_norm1 = np.mean(input_images_test_norm1.reshape(-1, 3, 3), axis=(-2, -1))
        # reshape
        RF_input_images_test_norm2 = RF_input_images_test_norm1.reshape(input_images_test_norm0.shape[0],
                                                                        input_images_test_norm0.shape[1],
                                                                        input_images_test_norm0.shape[2])
        # 若考虑气象协变量Rg_daily，PotRad_uStar_daily，Tair_f_daily，VPD_f_daily
        # RF_input_images_test_norm3 = np.concatenate((RF_input_images_test_norm2, input_covariate_test_norm0[:, :, 1:5]),
        #                                             axis=2)
        RF_input_images_test_norm3 = RF_input_images_test_norm2
        # reshape
        RF_input_images_test_norm4 = RF_input_images_test_norm3.reshape(
            RF_input_images_test_norm3.shape[0] * RF_input_images_test_norm3.shape[1],
            RF_input_images_test_norm3.shape[2])
        RF_input_images_test_norm4.shape
        # 对自变量值为-9999的数据进行处理
        RF_test_valid_indices = np.where((RF_input_images_test_norm4 != -9999).all(axis=1))
        RF_input_images_test_norm4_clean = RF_input_images_test_norm4[RF_test_valid_indices]
        RF_testy = RF_testy.reshape(RF_testy.shape[0] * RF_testy.shape[1], 3)
        RF_testy_clean = RF_testy[RF_test_valid_indices]
        test_sites_array_reshape = test_sites_array.reshape(test_sites_array.shape[0] * test_sites_array.shape[1],
            test_sites_array.shape[2])
        RF_test_sites_array = test_sites_array_reshape[RF_test_valid_indices]
        # print(RF_input_images_train_norm4_clean[:,:9])
        # 随机森林模型 _0
        # RFmodel = RandomForestRegressor(n_estimators=180, max_depth=6,
        #                                 min_samples_leaf=6,
        #                                 min_samples_split=7,
        #                                 max_leaf_nodes=41,
        #                                 n_jobs=-1, random_state=90)

        # # 随机森林模型 _1
        # RFmodel = RandomForestRegressor(n_estimators=180,
        #                                 max_depth=5,
        #                                 min_samples_leaf=4,
        #                                 min_samples_split=10,
        #                                 max_leaf_nodes=25,
        #                                 n_jobs=-1,
        #                                 random_state=90)
        # # 随机森林模型 _2
        # RFmodel = RandomForestRegressor(n_estimators=100,
        #                                 max_depth=5,
        #                                 min_samples_leaf=4,
        #                                 min_samples_split=10,
        #                                 max_leaf_nodes=25,
        #                                 n_jobs=-1,
        #                                 random_state=90)
        # 随机森林模型 _3
        RFmodel = RandomForestRegressor(n_estimators=100,
                                        min_samples_leaf=1,
                                        min_samples_split=2,
                                        n_jobs=-1,
                                        random_state=90)
        # [0,1,2,3,4,5,6,9,10,11,12,13,14]
        RF_trainy_input = RF_trainy_clean[:,0:1].ravel()
        RF_testy_input = RF_testy_clean[:,0:1].ravel()
        # 没有soil water
        RFmodel.fit(RF_input_images_train_norm4_clean[:,:11], RF_trainy_input)
        # r_score = RFmodel.score(RF_input_images_test_norm4_clean[:, :], RF_testy_input)
        # print(r_score)
        y_pred = RFmodel.predict(RF_input_images_test_norm4_clean[:, :11])
        y_test = RF_testy_input
        print(y_pred.shape)
        # 将testy转换为float类型
        Y_test_filtered.append(y_test)
        Y_pred_filtered.append(y_pred)
        RF_test_sites.append(RF_test_sites_array)
    Y_pred_filtered_array = np.concatenate(Y_pred_filtered, axis=0)
    Y_test_filtered_array = np.concatenate(Y_test_filtered, axis=0)
    RF_test_sites_array = np.concatenate(RF_test_sites, axis=0)
    # 将交叉验证的结果保存下来进行验证分析
    # results_variation_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_estimate_result/'
    np.save(results_variation_path + 'RF_RL_NEE_predict_soil.npy', Y_pred_filtered_array)
    np.save(results_variation_path + 'RF_RL_NEE_true_soil.npy', Y_test_filtered_array)
    np.save(results_variation_path + 'RF_RL_NEE_site_soil.npy', RF_test_sites_array)
    # all_y = np.array([Y_test_filtered_array, Y_pred_filtered_array])
    # fig, ax = plt.subplots(figsize=(8,8))
    # # Define the titles for each subgraph
    # # subgraph_titles = ['NEE(gC$\mathregular{m^{-2}}$$\mathregular{d^{-1}}$)']
    # # Plot the point density for each predicted value
    #
    # y_test_i = all_y[0, :]
    # y_pred_i = all_y[1, :]
    # # Filter out the data not equal to -9999.0
    # indices = np.where(y_test_i != -9999.0)
    # y_test_plot_i = np.array(y_test_i[indices], dtype=float)
    # y_test_plot_i = np.around(y_test_plot_i, decimals=5)
    #
    # y_pred_plot_i = np.array(y_pred_i[indices], dtype=float)
    # y_pred_plot_i = np.around(y_pred_plot_i, decimals=5)
    #
    # # Calculate the point density for the scatter plot
    # xy = np.vstack([y_test_plot_i, y_pred_plot_i]).astype(float)
    # z = gaussian_kde(xy)(xy)
    #
    # # Sort the points by density
    # idx = z.argsort()
    # x, y, z = y_test_plot_i[idx], y_pred_plot_i[idx], z[idx]
    #
    # # Plot the density scatter plot
    # sa = ax.scatter(x, y, c=z, s=50, cmap='jet', marker='.')
    #
    # # Calculate RMSE, CC, and n values
    # rmse = np.sqrt(mean_squared_error(y_test_plot_i, y_pred_plot_i))
    # cc = np.corrcoef(y_test_plot_i, y_pred_plot_i)[0, 1]
    # r2 = r2_score(y_test_plot_i, y_pred_plot_i)
    # n = len(y_test_plot_i)
    # bias = mean(y_test_plot_i - y_pred_plot_i)
    #
    # # Add text annotation
    # text = "RMSE = {:.2f}\nCC = {:.2f}\nBias = {:.2f}\nn = {}".format(rmse, cc, bias, n)
    # # text = "RMSE = {:.2f}\nR\u00b2 = {:.2f}\nn = {}".format(rmse,r2, n)
    # ax.annotate(text, xy=(0.05, 0.68), xycoords='axes fraction', color='black', fontsize=10,
    #             bbox=dict(facecolor='none', edgecolor='red'))
    #
    # # Calculate the coefficients of the linear fit
    # # Assuming you have defined get_regression_line function elsewhere
    # # regression_line = get_regression_line(y_test_plot_i, y_pred_plot_i,
    # #                                       data_range=(int(y_test_plot_i.min()), int(y_test_plot_i.max())))
    # # ax.plot(regression_line, 'r-', lw=2)
    #
    # # Set the title of the subgraph
    # ax.set_title('RECO(gC$\mathregular{m^{-2}}$$\mathregular{d^{-1}}$)', fontsize=12)
    #
    # # Set the x-axis and y-axis limits
    # min_val = min(min(y_test_plot_i), min(y_test_plot_i))
    # max_val = max(max(y_test_plot_i), max(y_test_plot_i))
    # ax.set_xlim([min_val, max_val])
    # ax.set_ylim([min_val, max_val])
    #
    # # Add a red diagonal line
    # ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    # ax.set_aspect('equal')
    # ax.set_xlabel('Y_true', fontsize=12)
    # ax.set_ylabel('Y_pred', fontsize=12)
    #
    # # Print RMSE and R2
    # print(f"Rmse_mean_cross: {rmse:.2f}")
    # print(f"r2_mean_cross: {r2:.2f}")
    # # plot_title = 'Density Plots of ' + 'using reflectance,lai/fpar and meteorology by RF model'
    # # plot_title = 'Density Plots of ' + 'using reflectance and meteorology by RF model'
    # # plot_title = 'Density Plots of ' + 'using reflectance and lai/fpar variables by RF model'
    # # plot_title = 'Density Plots of ' + 'using reflectance variables by RF model'
    # # plt.suptitle(plot_title, fontsize=15)
    # # Add a colorbar
    # plt.colorbar(sa)
    # plt.show()
