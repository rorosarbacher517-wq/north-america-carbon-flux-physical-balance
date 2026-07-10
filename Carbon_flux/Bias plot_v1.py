import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
# # 创建数据框
# data = {
#     'Ground NEE': [-3.65332572, -3.65332464, -3.653320968, -3.65316048, -3.65284296,
#                    -3.6526896, -3.65265936, -3.6526572, -3.652623072, -3.65260104,
#                    -3.65247144, -3.65243256, -3.65238504, -3.65238288, -3.65233752,
#                    -3.65227704, -3.6521928, -3.652177248, -3.652170336, -3.65216472,
#                    -3.65215392, -3.65208912, -3.65198328, -3.6518904],
#     'DL_NEE_RLM': [3.546815833, 2.109093106, 1.556658447, 1.955525902, 1.634739599,
#                    4.160505915, 2.552623812, 3.045050431, 1.778334751, 2.750478006,
#                    1.614234584, 2.821746825, 1.427808567, 1.466519147, 2.672644465,
#                    1.805558967, 1.521497219, 3.113505754, 1.141297257, 2.021465324,
#                    1.935787391, 2.745169164, 1.860380788, 1.267161015],
#     'DL_NEE_no_physical_RLM': [3.459175749, 2.547898329, 2.827917337, 1.735035804,
#                                 2.768967828, 3.722531179, 3.036541227, 2.528315593,
#                                 1.354286566, 3.349577493, 1.193243878, 2.854537725,
#                                 1.806175514, 1.562685519, 2.00040981, 1.4518476,
#                                 1.150948255, 3.616032977, 0.993632948, 2.723172568,
#                                 1.635359, 2.197683097, 2.101273794, 1.519532803],
#     'XGB_NEE_RLM': [2.251123948, 3.736019394, 1.582758605, 2.387747792, 2.002835831,
#                     3.076361978, 2.770188573, 1.861671258, 1.191146746, 2.318086899,
#                     0.696453946, 3.392509579, 2.396553441, 1.260443717, 2.071244328,
#                     2.19859724, 2.831110268, 3.088028822, 1.815816796, 2.956078671,
#                     2.094854544, 2.813699187, 2.701397795, 1.620587233],
#     'RF_NEE_RLM': [2.640576139, 3.517277038, 1.96824851, 2.590161353, 2.637547831,
#                    3.419323761, 2.774403447, 2.699083922, 1.441464265, 2.203173976,
#                    1.468115819, 2.161743368, 2.633812432, 1.704958082, 2.204010654,
#                    1.698068175, 2.69015351, 2.931243076, 0.796738042, 2.799849118,
#                    2.457289171, 1.579355592, 3.281985417, 1.353390654]
# }
#
# df = pd.DataFrame(data)
#
# # 根据Ground NEE排序
# df = df.sort_values(by='Ground NEE')
#
# # 划分区间
# bins = np.linspace(df['Ground NEE'].min(), df['Ground NEE'].max(), num=5)
# df['NEE_bins'] = pd.cut(df['Ground NEE'], bins)
#
# # 计算均值
# mean_values = df.groupby('NEE_bins').mean()
#
# # 可视化
# plt.plot(mean_values.index.astype(str), mean_values['DL_NEE_RLM'], label='DL_NEE_RLM')
# plt.plot(mean_values.index.astype(str), mean_values['DL_NEE_no_physical_RLM'], label='DL_NEE_no_physical_RLM')
# plt.plot(mean_values.index.astype(str), mean_values['XGB_NEE_RLM'], label='XGB_NEE_RLM')
# plt.plot(mean_values.index.astype(str), mean_values['RF_NEE_RLM'], label='RF_NEE_RLM')
#
# plt.title('Model Mean Values by Ground NEE Bins')
# plt.xlabel('Ground NEE Bins')
# plt.ylabel('Mean Values')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# ## real,pred = y_test_plot_i, y_pred_plot_i
# def get_regression_line(real, pred, data_range=(0, 110)):
#     # 拟合（若换MK，自行操作）最小二乘
#     def slope(xs, ys):
#         m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
#         b = mean(ys) - m * mean(xs)
#         return m, b
#
def interpolation_quality(DL_Y_predict,DL_Y_true,inter_mask,DL_y_qa,RF_NEE_predict,RF_GPP_predict,RF_RECO_predict,is_inter = 0):
    # x:插值与不插值；y:good quality 和other
    DL_mask = DL_Y_true[:, :, 2] == -9999
    inter_mask_condition = inter_mask[:, :, 0, 1, 1] == -9999
    DL_y_qa_mask = (DL_y_qa[:, :, 3] < 0) | (DL_y_qa[:, :, 3] > 0.2)
    if is_inter == 0:
        combined_mask = DL_mask
    # x:不插值；y:good quality 和other
    elif is_inter == 1:
        combined_mask = np.logical_or(inter_mask_condition, DL_mask)
    # x:不插值；y:good quality
    elif is_inter == 2:
        combined_mask = np.logical_or.reduce([inter_mask_condition,DL_y_qa_mask, DL_mask])
    # 三个值：深度学习预测的NEE
    DL_NEE = DL_Y_predict[:, :, 0][~combined_mask]  # 64425
    # 第四个值：深度学习预测的RECO-CPP 不等于-9999
    DL_GPP = DL_Y_predict[:, :, 1][~combined_mask]
    DL_RECO = DL_Y_predict[:, :, 2][~combined_mask]
    # 第一个值：地面站点的NEE
    Ground_NEE = DL_Y_true[:, :, 0][~combined_mask]  # 64425
    # 第二个值：地面站点的RECO-GPP
    # Ground_RECO_GPP = RF_Y_true_array[:, 2] - RF_Y_true_array[:, 1] # 64425
    Ground_GPP = DL_Y_true[:, :, 1][~combined_mask]
    Ground_RECO = DL_Y_true[:, :, 2][~combined_mask]
    # Flatten the 2D indices to 1D array
    # Obtain the indices of ~DL_mask
    index1 = np.argwhere(~DL_mask)
    # Use the indices from mask1 to select the indices from mask2
    index2 = np.argwhere(~combined_mask[index1[:, 0], index1[:, 1]].astype(bool))
    # Flatten the 2D indices to 1D array
    index2_flattened = index2.ravel()
    # # # 第五个值 单个随机森林预测的 NEE
    RF_NEE = RF_NEE_predict[index2_flattened]
    # 第六个值： 单个随机森林预测的RECO CPP
    RF_GPP = RF_GPP_predict[index2_flattened]
    RF_RECO = RF_RECO_predict[index2_flattened]
    return DL_NEE,DL_GPP,DL_RECO,Ground_NEE,Ground_GPP,Ground_RECO,RF_NEE, RF_GPP, RF_RECO

#
if __name__ == "__main__":
    # download the data
    inputdata_path = './data/sites_results/'
    # inputdata_path = './data/Sites_results_copy/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RL_predict_3.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RL_true_3.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RL_sites_3.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RL_x_mask_3.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RL_y_qa_3.npy', allow_pickle=True)
    # DL_Y_true[:,:,1]
    DL_Y_predict_no_physical = np.load(inputdata_path + 'DL_RL_predict_no_physical_3.npy', allow_pickle=True)
    DL_Y_true_no_physical = np.load(inputdata_path + 'DL_RL_true_no_physical_3.npy', allow_pickle=True)
    DL_All_true_no_physical = np.load(inputdata_path + 'DL_RL_sites_no_physical_3.npy', allow_pickle=True)
    inter_mask_no_physical = np.load(inputdata_path + 'DL_RL_x_mask_no_physical_3.npy', allow_pickle=True)
    DL_y_qa_no_physical = np.load(inputdata_path + 'DL_RL_y_qa_no_physical_3.npy', allow_pickle=True)

    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RL_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RL_NEE_true_3.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RL_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RL_GPP_true_3.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RL_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RL_RECO_true_3.npy', allow_pickle=True)

    # XGB three
    XGB_NEE_predict = np.load(inputdata_path + 'XGB_RL_NEE_predict_3.npy', allow_pickle=True)
    XGB_NEE_true = np.load(inputdata_path + 'XGB_RL_NEE_true_3.npy', allow_pickle=True)

    XGB_GPP_predict = np.load(inputdata_path + 'XGB_RL_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true = np.load(inputdata_path + 'XGB_RL_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict = np.load(inputdata_path + 'XGB_RL_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true = np.load(inputdata_path + 'XGB_RL_RECO_true_3.npy', allow_pickle=True)

    # RF 没有气象数据
    DL_NEE_0, DL_GPP_0, DL_RECO_0, Ground_NEE_0, Ground_GPP_0, Ground_RECO_0, RF_NEE_0, RF_GPP_0, RF_RECO_0 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=0)
    DL_NEE_1, DL_GPP_1, DL_RECO_1, Ground_NEE_1, Ground_GPP_1, Ground_RECO_1, RF_NEE_1, RF_GPP_1, RF_RECO_1 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=1)
    DL_NEE_2, DL_GPP_2, DL_RECO_2, Ground_NEE_2, Ground_GPP_2, Ground_RECO_2, RF_NEE_2, RF_GPP_2, RF_RECO_2 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=2)

    # XGB 没有气象数据
    DL_NEE_0_no_physical, DL_GPP_0_no_physical, DL_RECO_0_no_physical, Ground_NEE_0, Ground_GPP_0, Ground_RECO_0, XGB_NEE_0, XGB_GPP_0, XGB_RECO_0 = interpolation_quality(
        DL_Y_predict_no_physical, DL_Y_true_no_physical, inter_mask_no_physical, DL_y_qa_no_physical, XGB_NEE_predict,
        XGB_GPP_predict, XGB_RECO_predict, is_inter=0)
    DL_NEE_1_no_physical, DL_GPP_1_no_physical, DL_RECO_1_no_physical, Ground_NEE_1, Ground_GPP_1, Ground_RECO_1, XGB_NEE_1, XGB_GPP_1, XGB_RECO_1 = interpolation_quality(
        DL_Y_predict_no_physical, DL_Y_true_no_physical, inter_mask_no_physical, DL_y_qa_no_physical, XGB_NEE_predict,
        XGB_GPP_predict, XGB_RECO_predict, is_inter=1)
    DL_NEE_2_no_physical, DL_GPP_2_no_physical, DL_RECO_2_no_physical, Ground_NEE_2, Ground_GPP_2, Ground_RECO_2, XGB_NEE_2, XGB_GPP_2, XGB_RECO_2 = interpolation_quality(
        DL_Y_predict_no_physical, DL_Y_true_no_physical, inter_mask_no_physical, DL_y_qa_no_physical, XGB_NEE_predict,
        XGB_GPP_predict, XGB_RECO_predict, is_inter=2)

    # linear dl
    DL_Y_predict_RLM = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    DL_Y_true_RLM = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    DL_All_true_RLM = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    inter_mask_RLM = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    DL_y_qa_RLM = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)

    DL_Y_predict_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_predict_no_physical_3.npy', allow_pickle=True)
    DL_Y_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_true_no_physical_3.npy', allow_pickle=True)
    DL_All_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_sites_no_physical_3.npy', allow_pickle=True)
    inter_mask_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_x_mask_no_physical_3.npy', allow_pickle=True)
    DL_y_qa_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_y_qa_no_physical_3.npy', allow_pickle=True)

    # RF three
    RF_NEE_predict_RLM = np.load(inputdata_path + 'RF_RLM_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true_RLM = np.load(inputdata_path + 'RF_RLM_NEE_true_3.npy', allow_pickle=True)

    RF_GPP_predict_RLM = np.load(inputdata_path + 'RF_RLM_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true_RLM = np.load(inputdata_path + 'RF_RLM_GPP_true_3.npy', allow_pickle=True)

    RF_RECO_predict_RLM = np.load(inputdata_path + 'RF_RLM_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true_RLM = np.load(inputdata_path + 'RF_RLM_RECO_true_3.npy', allow_pickle=True)

    # XGB three
    XGB_NEE_predict_RLM = np.load(inputdata_path + 'XGB_RLM_NEE_predict_3.npy', allow_pickle=True)
    XGB_NEE_true_RLM = np.load(inputdata_path + 'XGB_RLM_NEE_true_3.npy', allow_pickle=True)
    #
    XGB_GPP_predict_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_true_3.npy', allow_pickle=True)

    # *******************************
    # DL 单独预测的 有气象数据
    DL_NEE_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_NEE_Single_2.npy', allow_pickle=True)
    DL_NEE_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_NEE_Single_2.npy', allow_pickle=True)
    #
    DL_GPP_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_GPP_Single_2.npy', allow_pickle=True)
    DL_GPP_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_GPP_Single_2.npy', allow_pickle=True)

    DL_RECO_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_RECO_Single_2.npy', allow_pickle=True)
    DL_RECO_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_RECO_Single_2.npy', allow_pickle=True)

    # 将数组的维数从三维转为一维
    DL_NEE_predict_RLM_Single_flattened = DL_NEE_predict_RLM_Single.reshape(-1)
    DL_NEE_true_RLM_Single_flattened = DL_NEE_true_RLM_Single.reshape(-1)

    DL_GPP_predict_RLM_Single_flattened = DL_GPP_predict_RLM_Single.reshape(-1)
    DL_GPP_true_RLM_Single_flattened = DL_GPP_true_RLM_Single.reshape(-1)

    DL_RECO_predict_RLM_Single_flattened = DL_RECO_predict_RLM_Single.reshape(-1)
    DL_RECO_true_RLM_Single_flattened = DL_RECO_true_RLM_Single.reshape(-1)

    # 假设两个数组已经变为一维
    NEE_RLM_mask = DL_NEE_true_RLM_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_NEE_predict_RLM_Single_filtered = DL_NEE_predict_RLM_Single_flattened[NEE_RLM_mask]
    DL_NEE_true_RLM_Single_filtered = DL_NEE_true_RLM_Single_flattened[NEE_RLM_mask]

    # 假设两个数组已经变为一维
    GPP_RLM_mask = DL_NEE_true_RLM_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_GPP_predict_RLM_Single_filtered = DL_GPP_predict_RLM_Single_flattened[GPP_RLM_mask]
    DL_GPP_true_RLM_Single_filtered = DL_GPP_true_RLM_Single_flattened[GPP_RLM_mask]

    # 假设两个数组已经变为一维
    RECO_RLM_mask = DL_RECO_true_RLM_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_RECO_predict_RLM_Single_filtered = DL_RECO_predict_RLM_Single_flattened[RECO_RLM_mask]
    DL_RECO_true_RLM_Single_filtered = DL_RECO_true_RLM_Single_flattened[RECO_RLM_mask]

    # *******************
    # DL 单独预测的 没有气象变量数据
    DL_NEE_predict_RL_Single = np.load(inputdata_path + 'DL_RL_predict_NEE_Single.npy', allow_pickle=True)
    DL_NEE_true_RL_Single = np.load(inputdata_path + 'DL_RL_true_NEE_Single.npy', allow_pickle=True)
    #
    DL_GPP_predict_RL_Single = np.load(inputdata_path + 'DL_RL_predict_GPP_Single.npy', allow_pickle=True)
    DL_GPP_true_RL_Single = np.load(inputdata_path + 'DL_RL_true_GPP_Single.npy', allow_pickle=True)

    DL_RECO_predict_RL_Single = np.load(inputdata_path + 'DL_RL_predict_RECO_Single.npy', allow_pickle=True)
    DL_RECO_true_RL_Single = np.load(inputdata_path + 'DL_RL_true_RECO_Single.npy', allow_pickle=True)

    # 将数组的维数从三维转为一维
    DL_NEE_predict_RL_Single_flattened = DL_NEE_predict_RL_Single.reshape(-1)
    DL_NEE_true_RL_Single_flattened = DL_NEE_true_RL_Single.reshape(-1)

    DL_GPP_predict_RL_Single_flattened = DL_GPP_predict_RL_Single.reshape(-1)
    DL_GPP_true_RL_Single_flattened = DL_GPP_true_RL_Single.reshape(-1)

    DL_RECO_predict_RL_Single_flattened = DL_RECO_predict_RL_Single.reshape(-1)
    DL_RECO_true_RL_Single_flattened = DL_RECO_true_RL_Single.reshape(-1)

    # 假设两个数组已经变为一维
    NEE_RL_mask = DL_NEE_true_RL_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_NEE_predict_RL_Single_filtered = DL_NEE_predict_RL_Single_flattened[NEE_RL_mask]
    DL_NEE_true_RL_Single_filtered = DL_NEE_true_RL_Single_flattened[NEE_RL_mask]

    # 假设两个数组已经变为一维
    GPP_RL_mask = DL_NEE_true_RLM_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_GPP_predict_RL_Single_filtered = DL_GPP_predict_RL_Single_flattened[GPP_RL_mask]
    DL_GPP_true_RL_Single_filtered = DL_GPP_true_RL_Single_flattened[GPP_RL_mask]

    # 假设两个数组已经变为一维
    RECO_RL_mask = DL_RECO_true_RLM_Single_flattened != -9999  # 创建布尔掩码，True 表示保留的元素
    # 仅保留对应位置的有效值
    DL_RECO_predict_RL_Single_filtered = DL_RECO_predict_RL_Single_flattened[RECO_RL_mask]
    DL_RECO_true_RL_Single_filtered = DL_RECO_true_RL_Single_flattened[RECO_RL_mask]

    # print(min(XGB_GPP_predict),min(XGB_GPP_predict),min(XGB_GPP_predict_RLM),min(XGB_GPP_predict_RLM))
    # print(min(XGB_RECO_predict), min(XGB_RECO_predict), min(XGB_RECO_predict_RLM), min(XGB_RECO_predict_RLM))
    #     # RF 有气象变量
    DL_NEE_0_RLM, DL_GPP_0_RLM, DL_RECO_0_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, RF_NEE_0_RLM, RF_GPP_0_RLM, RF_RECO_0_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=0)
    DL_NEE_1_RLM, DL_GPP_1_RLM, DL_RECO_1_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, RF_NEE_1_RLM, RF_GPP_1_RLM, RF_RECO_1_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=1)
    DL_NEE_2_RLM, DL_GPP_2_RLM, DL_RECO_2_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, RF_NEE_2_RLM, RF_GPP_2_RLM, RF_RECO_2_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=2)

    # 第XGB 有气象变量
    DL_NEE_0_no_physical_RLM, DL_GPP_0_no_physical_RLM, DL_RECO_0_no_physical_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, XGB_NEE_0_RLM, XGB_GPP_0_RLM, XGB_RECO_0_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=0)
    DL_NEE_1_no_physical_RLM, DL_GPP_1_no_physical_RLM, DL_RECO_1_no_physical_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, XGB_NEE_1_RLM, XGB_GPP_1_RLM, XGB_RECO_1_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=1)
    DL_NEE_2_no_physical_RLM, DL_GPP_2_no_physical_RLM, DL_RECO_2_no_physical_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, XGB_NEE_2_RLM, XGB_GPP_2_RLM, XGB_RECO_2_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=2)

    # DL RLM单独预测的碳通量 有气象变量
    # 第一组值:插值与不插值的一起
    DL_NEE_0_no_physical_RLM, DL_GPP_0_no_physical_RLM, DL_RECO_0_no_physical_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, DL_NEE_0_RLM_Single, DL_GPP_0_RLM_Single, DL_RECO_0_RLM_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RLM_Single_filtered, DL_GPP_predict_RLM_Single_filtered, DL_RECO_predict_RLM_Single_filtered,
        is_inter=0)
    DL_NEE_1_no_physical_RLM, DL_GPP_1_no_physical_RLM, DL_RECO_1_no_physical_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, DL_NEE_1_RLM_Single, DL_GPP_1_RLM_Single, DL_RECO_1_RLM_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RLM_Single_filtered, DL_GPP_predict_RLM_Single_filtered, DL_RECO_predict_RLM_Single_filtered,
        is_inter=1)
    DL_NEE_2_no_physical_RLM, DL_GPP_2_no_physical_RLM, DL_RECO_2_no_physical_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, DL_NEE_2_RLM_Single, DL_GPP_2_RLM_Single, DL_RECO_2_RLM_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RLM_Single_filtered, DL_GPP_predict_RLM_Single_filtered, DL_RECO_predict_RLM_Single_filtered,
        is_inter=2)

    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    DL_NEE_0_no_physical_RLM, DL_GPP_0_no_physical_RLM, DL_RECO_0_no_physical_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, DL_NEE_0_RL_Single, DL_GPP_0_RL_Single, DL_RECO_0_RL_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,
        is_inter=0)
    DL_NEE_1_no_physical_RLM, DL_GPP_1_no_physical_RLM, DL_RECO_1_no_physical_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, DL_NEE_1_RL_Single, DL_GPP_1_RL_Single, DL_RECO_1_RL_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,
        is_inter=1)
    DL_NEE_2_no_physical_RLM, DL_GPP_2_no_physical_RLM, DL_RECO_2_no_physical_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, DL_NEE_2_RL_Single, DL_GPP_2_RL_Single, DL_RECO_2_RL_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,
        is_inter=2)

    # 假设 df_predictions 和 df_ground_truth 已被正确定义
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.font_manager import FontProperties
    # 假设 df_predictions 和 df_ground_truth 已被正确定义
    # 例如
    ground_truth_data = {
        'Ground_NEE_0': Ground_NEE_0,
        'Ground_GPP_0': Ground_GPP_0,
        'Ground_RECO_0': Ground_RECO_0
    }

    pred_data = {
        'DL_NEE_0_RLM': DL_NEE_0_RLM,
        'DL_NEE_0_no_physical_RLM': DL_NEE_0_no_physical_RLM,
        'DL_NEE_0_RLM_Single':DL_NEE_0_RLM_Single,
        'XGB_NEE_0_RLM': XGB_NEE_0_RLM,
        'RF_NEE_0_RLM': RF_NEE_0_RLM,
        'DL_NEE_0': DL_NEE_0,
        'DL_NEE_0_no_physical': DL_NEE_0_no_physical,
        'DL_NEE_0_RL_Single': DL_NEE_0_RL_Single,
        'XGB_NEE_0': XGB_NEE_0,
        'RF_NEE_0': RF_NEE_0,

        'DL_GPP_0_RLM': DL_GPP_0_RLM,
        'DL_GPP_0_no_physical_RLM': DL_GPP_0_no_physical_RLM,
        'DL_GPP_0_RLM_Single': DL_GPP_0_RLM_Single,
        'XGB_GPP_0_RLM': XGB_GPP_0_RLM,
        'RF_GPP_0_RLM': RF_GPP_0_RLM,
        'DL_GPP_0': DL_GPP_0,
        'DL_GPP_0_no_physical': DL_GPP_0_no_physical,
        'DL_GPP_0_RL_Single': DL_GPP_0_RL_Single,
        'XGB_GPP_0': XGB_GPP_0,
        'RF_GPP_0': RF_GPP_0,

        'DL_RECO_0_RLM': DL_RECO_0_RLM,
        'DL_RECO_0_no_physical_RLM': DL_RECO_0_no_physical_RLM,
        'DL_RECO_0_RLM_Single': DL_RECO_0_RLM_Single,
        'XGB_RECO_0_RLM': XGB_RECO_0_RLM,
        'RF_RECO_0_RLM': RF_RECO_0_RLM,
        'DL_RECO_0': DL_RECO_0,
        'DL_RECO_0_no_physical': DL_RECO_0_no_physical,
        'DL_RECO_0_RL_Single': DL_RECO_0_RL_Single,
        'XGB_RECO_0': XGB_RECO_0,
        'RF_RECO_0': RF_RECO_0,
    }
    df_ground_truth = pd.DataFrame(ground_truth_data)
    df_predictions = pd.DataFrame(pred_data)
    biases_with_met = pd.DataFrame({
        'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_0_RLM_Single':df_predictions['DL_NEE_0_RLM_Single'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
        'DL_GPP_0_RLM_Single': df_predictions['DL_GPP_0_RLM_Single'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
        'DL_RECO_0_RLM_Single': df_predictions['DL_RECO_0_RLM_Single'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    })

    biases_without_met = pd.DataFrame({
        'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
        'DL_NEE_0_RL_Single': df_predictions['DL_NEE_0_RL_Single'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_0_RL_Single': df_predictions['DL_GPP_0_RL_Single'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
        'DL_RECO_0_RL_Single': df_predictions['DL_RECO_0_RL_Single'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    })

    data = {
        'Ground NEE_RLM': df_ground_truth['Ground_NEE_0'],
        'DL_NEE_RLM': biases_with_met['DL_NEE_RLM'],
        'DL_NEE_no_physical_RLM': biases_with_met['DL_NEE_no_physical_RLM'],
        'DL_NEE_0_RLM_Single':biases_with_met['DL_NEE_0_RLM_Single'],
        'XGB_NEE_RLM': biases_with_met['XGB_NEE_RLM'],
        'RF_NEE_RLM': biases_with_met['RF_NEE_RLM'],

        'Ground GPP_RLM': df_ground_truth['Ground_GPP_0'],
        'DL_GPP_RLM': biases_with_met['DL_GPP_RLM'],
        'DL_GPP_no_physical_RLM': biases_with_met['DL_GPP_no_physical_RLM'],
        'DL_GPP_0_RLM_Single': biases_with_met['DL_GPP_0_RLM_Single'],
        'XGB_GPP_RLM': biases_with_met['XGB_GPP_RLM'],
        'RF_GPP_RLM': biases_with_met['RF_GPP_RLM'],

        'Ground RECO_RLM': df_ground_truth['Ground_RECO_0'],
        'DL_RECO_RLM': biases_with_met['DL_RECO_RLM'],
        'DL_RECO_no_physical_RLM': biases_with_met['DL_RECO_no_physical_RLM'],
        'DL_RECO_0_RLM_Single': biases_with_met['DL_RECO_0_RLM_Single'],
        'XGB_RECO_RLM': biases_with_met['XGB_RECO_RLM'],
        'RF_RECO_RLM': biases_with_met['RF_RECO_RLM'],

        'Ground NEE': df_ground_truth['Ground_NEE_0'],
        'DL_NEE': biases_without_met['DL_NEE'],
        'DL_NEE_no_physical': biases_without_met['DL_NEE_no_physical'],
        'DL_NEE_0_RL_Single': biases_without_met['DL_NEE_0_RL_Single'],
        'XGB_NEE': biases_without_met['XGB_NEE'],
        'RF_NEE': biases_without_met['RF_NEE'],

        'Ground GPP': df_ground_truth['Ground_GPP_0'],
        'DL_GPP': biases_without_met['DL_GPP'],
        'DL_GPP_no_physical': biases_without_met['DL_GPP_no_physical'],
        'DL_GPP_0_RL_Single': biases_without_met['DL_GPP_0_RL_Single'],
        'XGB_GPP': biases_without_met['XGB_GPP'],
        'RF_GPP': biases_without_met['RF_GPP'],

        'Ground RECO': df_ground_truth['Ground_RECO_0'],
        'DL_RECO': biases_without_met['DL_RECO'],
        'DL_RECO_no_physical': biases_without_met['DL_RECO_no_physical'],
        'DL_RECO_0_RL_Single': biases_without_met['DL_RECO_0_RL_Single'],
        'XGB_RECO': biases_without_met['XGB_RECO'],
        'RF_RECO': biases_without_met['RF_RECO'],
    }

    # 创建数据框
    # 创建数据框
    df = pd.DataFrame(data)
    plt.rcParams['font.family'] = 'Arial'
    # model_subtitle = ['Physics-aware\ntransformer', 'XGBoost', 'RF']
    model_subtitle = ['DL_physic', 'DL_no_physic', 'DL_single']
    # 定义颜色
    colors1 = ['red', 'green', 'blue']  # Colors for the three models
    colors2 = ['#F36B6F', '#37AB78', '#8EC4D4']  # Lighter colors

    # Combine colors1 and colors2
    colors = []
    for c1, c2 in zip(colors1, colors2):
        rgb1 = mcolors.to_rgb(c1)
        rgb2 = mcolors.to_rgb(c2)
        new_rgb = [(r1 + r2) / 2 for r1, r2 in zip(rgb1, rgb2)]
        colors.append(mcolors.to_hex(new_rgb))


    # 定义绘图函数
    def plot_model_biases(ax, index, ground_col, model_cols, title):
        subdf = df[[ground_col] + model_cols]
        df_sorted = subdf.sort_values(by=ground_col)

        bins = np.linspace(df_sorted[ground_col].min(), df_sorted[ground_col].max(), num=10)
        df_sorted['Bins'] = pd.cut(df_sorted[ground_col], bins)

        mean_values = df_sorted.groupby('Bins')[model_cols].mean()

        # 只取右区间值
        right_edges = [bin.right for bin in mean_values.index]

        # Plot each model with corresponding color
        for model, color in zip(model_cols, colors):
            ax.plot(right_edges, mean_values[model], marker='o', label=model, color=color)

        # 添加虚线
        ax.axhline(0, color='gray', linestyle='--')

        if index < 3:
            ax.set_title(title, fontsize=24)
        ax.set_xticks(right_edges)  # 只显示右区间值
        ax.set_xticklabels([f"{edge:.0f}" for edge in right_edges], rotation=0)  # 旋转以便清晰

        # # 设置y-label和x-label
        # if index % 3 == 0:  # 第一列
        #     if index < 3:  # 第一行
        #         ax.set_ylabel('Mean Bias (input with meteorological data)', fontsize=16)
        #     else:  # 第二行
        #         ax.set_ylabel('Mean Bias (input without meteorological data)', fontsize=16)

        # if index >= 3:  # 第一行
        ax.set_xlabel('Flux tower bins', fontsize=23)
        ax.set_ylabel('Average Bias', fontsize=23)

        ax.legend(model_subtitle, fontsize=20)

        # 设置横纵坐标刻度为整数
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=20)
        # 设置坐标轴刻度字体大小
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(20)

    # 创建子图
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    groups = [
        ('Ground NEE', ['DL_NEE_RLM', 'DL_NEE_no_physical_RLM', 'DL_NEE_0_RLM_Single'], 'NEE'),
        ('Ground GPP', ['DL_GPP_RLM', 'DL_GPP_no_physical_RLM', 'DL_GPP_0_RLM_Single'], 'GPP'),
        ('Ground RECO', ['DL_RECO_RLM', 'DL_RECO_no_physical_RLM', 'DL_RECO_0_RLM_Single'], 'RECO'),
        ('Ground NEE', ['DL_NEE', 'DL_NEE_no_physical', 'DL_NEE_0_RL_Single'], 'NEE without Meteorological Data'),
        ('Ground GPP', ['DL_GPP', 'DL_GPP_no_physical', 'DL_GPP_0_RL_Single'], 'GPP without Meteorological Data'),
        ('Ground RECO', ['DL_RECO', 'DL_RECO_no_physical', 'DL_RECO_0_RL_Single'], 'RECO without Meteorological Data')
    ]

    for idx, (ax, (ground_col, model_cols, title)) in enumerate(zip(axs.flatten(), groups)):
        plot_model_biases(ax, idx, ground_col, model_cols, title)

    fig.subplots_adjust(right=0.9)

    fig.text(0.03, 0.75, '(a) Input with meteorological data', ha='center', va='center', rotation='vertical',
             fontsize=23, fontname='Arial')
    fig.text(0.03, 0.3, '(b) Input without meteorological data', ha='center', va='center', rotation='vertical',
             fontsize=23, fontname='Arial')
    # Adjust overall layout
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.95, wspace=0.2, hspace=0.25)
    # # 调整布局
    # plt.tight_layout()
    plt.show()
