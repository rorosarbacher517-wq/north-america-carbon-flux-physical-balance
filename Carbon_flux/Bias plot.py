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

    # 第一组值:插值与不插值的一起
    DL_NEE_0, DL_GPP_0, DL_RECO_0, Ground_NEE_0, Ground_GPP_0, Ground_RECO_0, RF_NEE_0, RF_GPP_0, RF_RECO_0 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=0)
    DL_NEE_1, DL_GPP_1, DL_RECO_1, Ground_NEE_1, Ground_GPP_1, Ground_RECO_1, RF_NEE_1, RF_GPP_1, RF_RECO_1 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=1)
    DL_NEE_2, DL_GPP_2, DL_RECO_2, Ground_NEE_2, Ground_GPP_2, Ground_RECO_2, RF_NEE_2, RF_GPP_2, RF_RECO_2 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, RF_NEE_predict, RF_GPP_predict, RF_RECO_predict, is_inter=2)

    # 第一组值:插值与不插值的一起
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

    XGB_GPP_predict_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_true_3.npy', allow_pickle=True)

    # 第一组值:插值与不插值的一起
    DL_NEE_0_RLM, DL_GPP_0_RLM, DL_RECO_0_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, RF_NEE_0_RLM, RF_GPP_0_RLM, RF_RECO_0_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=0)
    DL_NEE_1_RLM, DL_GPP_1_RLM, DL_RECO_1_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, RF_NEE_1_RLM, RF_GPP_1_RLM, RF_RECO_1_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=1)
    DL_NEE_2_RLM, DL_GPP_2_RLM, DL_RECO_2_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, RF_NEE_2_RLM, RF_GPP_2_RLM, RF_RECO_2_RLM = interpolation_quality(
        DL_Y_predict_RLM, DL_Y_true_RLM, inter_mask_RLM, DL_y_qa_RLM, RF_NEE_predict_RLM, RF_GPP_predict_RLM,
        RF_RECO_predict_RLM, is_inter=2)

    # 第一组值:插值与不插值的一起
    DL_NEE_0_no_physical_RLM, DL_GPP_0_no_physical_RLM, DL_RECO_0_no_physical_RLM, Ground_NEE_0_RLM, Ground_GPP_0_RLM, Ground_RECO_0_RLM, XGB_NEE_0_RLM, XGB_GPP_0_RLM, XGB_RECO_0_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=0)
    DL_NEE_1_no_physical_RLM, DL_GPP_1_no_physical_RLM, DL_RECO_1_no_physical_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, XGB_NEE_1_RLM, XGB_GPP_1_RLM, XGB_RECO_1_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=1)
    DL_NEE_2_no_physical_RLM, DL_GPP_2_no_physical_RLM, DL_RECO_2_no_physical_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, XGB_NEE_2_RLM, XGB_GPP_2_RLM, XGB_RECO_2_RLM = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        XGB_NEE_predict_RLM, XGB_GPP_predict_RLM, XGB_RECO_predict_RLM, is_inter=2)



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
        'XGB_NEE_0_RLM': XGB_NEE_0_RLM,
        'RF_NEE_0_RLM': RF_NEE_0_RLM,
        'DL_NEE_0': DL_NEE_0,
        'DL_NEE_0_no_physical': DL_NEE_0_no_physical,
        'XGB_NEE_0': XGB_NEE_0,
        'RF_NEE_0': RF_NEE_0,

        'DL_GPP_0_RLM': DL_GPP_0_RLM,
        'DL_GPP_0_no_physical_RLM': DL_GPP_0_no_physical_RLM,
        'XGB_GPP_0_RLM': XGB_GPP_0_RLM,
        'RF_GPP_0_RLM': RF_GPP_0_RLM,
        'DL_GPP_0': DL_GPP_0,
        'DL_GPP_0_no_physical': DL_GPP_0_no_physical,
        'XGB_GPP_0': XGB_GPP_0,
        'RF_GPP_0': RF_GPP_0,

        'DL_RECO_0_RLM': DL_RECO_0_RLM,
        'DL_RECO_0_no_physical_RLM': DL_RECO_0_no_physical_RLM,
        'XGB_RECO_0_RLM': XGB_RECO_0_RLM,
        'RF_RECO_0_RLM': RF_RECO_0_RLM,
        'DL_RECO_0': DL_RECO_0,
        'DL_RECO_0_no_physical': DL_RECO_0_no_physical,
        'XGB_RECO_0': XGB_RECO_0,
        'RF_RECO_0': RF_RECO_0,
    }
    df_ground_truth = pd.DataFrame(ground_truth_data)
    df_predictions = pd.DataFrame(pred_data)
    biases_with_met = pd.DataFrame({
        'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    })

    biases_without_met = pd.DataFrame({
        'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    })

    data = {
        'Ground NEE_RLM': df_ground_truth['Ground_NEE_0'],
        'DL_NEE_RLM': biases_with_met['DL_NEE_RLM'],
        'DL_NEE_no_physical_RLM': biases_with_met['DL_NEE_no_physical_RLM'],
        'XGB_NEE_RLM': biases_with_met['XGB_NEE_RLM'],
        'RF_NEE_RLM': biases_with_met['RF_NEE_RLM'],

        'Ground GPP_RLM': df_ground_truth['Ground_GPP_0'],
        'DL_GPP_RLM': biases_with_met['DL_GPP_RLM'],
        'DL_GPP_no_physical_RLM': biases_with_met['DL_GPP_no_physical_RLM'],
        'XGB_GPP_RLM': biases_with_met['XGB_GPP_RLM'],
        'RF_GPP_RLM': biases_with_met['RF_GPP_RLM'],

        'Ground RECO_RLM': df_ground_truth['Ground_RECO_0'],
        'DL_RECO_RLM': biases_with_met['DL_RECO_RLM'],
        'DL_RECO_no_physical_RLM': biases_with_met['DL_RECO_no_physical_RLM'],
        'XGB_RECO_RLM': biases_with_met['XGB_RECO_RLM'],
        'RF_RECO_RLM': biases_with_met['RF_RECO_RLM'],

        'Ground NEE': df_ground_truth['Ground_NEE_0'],
        'DL_NEE': biases_without_met['DL_NEE'],
        'DL_NEE_no_physical': biases_without_met['DL_NEE_no_physical'],
        'XGB_NEE': biases_without_met['XGB_NEE'],
        'RF_NEE': biases_without_met['RF_NEE'],

        'Ground GPP': df_ground_truth['Ground_GPP_0'],
        'DL_GPP': biases_without_met['DL_GPP'],
        'DL_GPP_no_physical': biases_without_met['DL_GPP_no_physical'],
        'XGB_GPP': biases_without_met['XGB_GPP'],
        'RF_GPP': biases_without_met['RF_GPP'],

        'Ground RECO': df_ground_truth['Ground_RECO_0'],
        'DL_RECO': biases_without_met['DL_RECO'],
        'DL_RECO_no_physical': biases_without_met['DL_RECO_no_physical'],
        'XGB_RECO': biases_without_met['XGB_RECO'],
        'RF_RECO': biases_without_met['RF_RECO'],
    }

    # 创建数据框
    # 创建数据框
    df = pd.DataFrame(data)
    plt.rcParams['font.family'] = 'Arial'
    model_subtitle = ['Physics-aware\ntransformer', 'XGBoost', 'RF']
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
        ('Ground NEE', ['DL_NEE_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM'], 'NEE'),
        ('Ground GPP', ['DL_GPP_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM'], 'GPP'),
        ('Ground RECO', ['DL_RECO_RLM', 'XGB_RECO_RLM', 'RF_RECO_RLM'], 'RECO'),
        ('Ground NEE', ['DL_NEE', 'XGB_NEE', 'RF_NEE'], 'NEE without Meteorological Data'),
        ('Ground GPP', ['DL_GPP', 'XGB_GPP', 'RF_GPP'], 'GPP without Meteorological Data'),
        ('Ground RECO', ['DL_RECO', 'XGB_RECO', 'RF_RECO'], 'RECO without Meteorological Data')
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

    # model_subtitle = ['Time series deep learning model\nwith physical constraint',
    #                   'Time series deep learning model\nwithout physical constraint', 'XGBoost', 'RF']
    #
    # # 定义绘图函数
    # def plot_model_biases(ax, index, ground_col, model_cols, title):
    #     subdf = df[[ground_col] + model_cols]
    #     df_sorted = subdf.sort_values(by=ground_col)
    #
    #     bins = np.linspace(df_sorted[ground_col].min(), df_sorted[ground_col].max(), num=10)
    #     df_sorted['Bins'] = pd.cut(df_sorted[ground_col], bins)
    #
    #     mean_values = df_sorted.groupby('Bins')[model_cols].mean()
    #
    #     # 只取右区间值
    #     right_edges = [bin.right for bin in mean_values.index]
    #
    #     for model in model_cols:
    #         ax.plot(right_edges, mean_values[model], marker='o', label=model)
    #
    #     # 添加虚线
    #     ax.axhline(0, color='gray', linestyle='--')
    #
    #     if index < 3:
    #         ax.set_title(title)
    #     ax.set_xticks(right_edges)  # 只显示右区间值
    #     ax.set_xticklabels([f"{edge:.2f}" for edge in right_edges], rotation=0)  # 旋转以便清晰
    #
    #     # 设置y-label和x-label
    #     if index % 3 == 0:  # 第一列
    #         if index < 3:  # 第一行
    #             ax.set_ylabel('Mean Bias (input with meteorological data)', fontsize=12)
    #         else:  # 第二行
    #             ax.set_ylabel('Mean Bias (input without meteorological data)', fontsize=12)
    #
    #     if index >= 3:  # 第一行
    #         ax.set_xlabel('Flux tower bins', fontsize=12)
    #
    #     ax.legend(model_subtitle)
    #
    #
    # # 创建子图
    # fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    #
    # groups = [
    #     ('Ground NEE', ['DL_NEE_RLM', 'DL_NEE_no_physical_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM'],
    #      'NEE'),
    #     ('Ground GPP', ['DL_GPP_RLM', 'DL_GPP_no_physical_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM'],
    #      'GPP'),
    #     ('Ground RECO', ['DL_RECO_RLM', 'DL_RECO_no_physical_RLM', 'XGB_RECO_RLM', 'RF_RECO_RLM'],
    #      'RECO'),
    #     ('Ground NEE', ['DL_NEE', 'DL_NEE_no_physical', 'XGB_NEE', 'RF_NEE'], 'NEE without Meteorological Data'),
    #     ('Ground GPP', ['DL_GPP', 'DL_GPP_no_physical', 'XGB_GPP', 'RF_GPP'], 'GPP without Meteorological Data'),
    #     ('Ground RECO', ['DL_RECO', 'DL_RECO_no_physical', 'XGB_RECO', 'RF_RECO'], 'RECO without Meteorological Data')
    # ]
    #
    # for idx, (ax, (ground_col, model_cols, title)) in enumerate(zip(axs.flatten(), groups)):
    #     plot_model_biases(ax, idx, ground_col, model_cols, title)
    #
    # # 调整布局
    # plt.tight_layout()
    # plt.show()

    # df = pd.DataFrame(data)
    # plt.rcParams['font.family'] = 'Arial'
    # legend_name = ['Time series deep learning model\nwith physical constraint',
    #                   'Time series deep learning model\nwithout physical constraint', 'XGBoost',
    #                   'RF']  # 'CFR without physical constraints',
    #
    # # 定义绘图函数
    # def plot_model_biases(ax, ground_col, model_cols, title):
    #     # 创建一个子集
    #     subdf = df[[ground_col] + model_cols]
    #
    #     # 按照 Ground NEE 排序
    #     df_sorted = subdf.sort_values(by=ground_col)
    #
    #     # 分区间
    #     bins = np.linspace(df_sorted[ground_col].min(), df_sorted[ground_col].max(), num=10)
    #     df_sorted['Bins'] = pd.cut(df_sorted[ground_col], bins)
    #
    #     # 计算每个区间各个模型的均值
    #     mean_values = df_sorted.groupby('Bins')[model_cols].mean()
    #
    #     # 循环绘制每个模型的折线图
    #     for model in model_cols:
    #         ax.plot(mean_values.index.astype(str), mean_values[model], marker='o', label=model)
    #
    #     ax.set_title(title)
    #     ax.set_xlabel('Ground Value Bins')
    #     ax.set_ylabel('Mean Bias')
    #     ax.legend()
    #
    # # 创建子图
    # fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #
    # # 不同模型和变量的组别
    # groups = [
    #     ('Ground NEE', ['DL_NEE_RLM', 'DL_NEE_no_physical_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM'],
    #      'NEE with Meteorological Data'),
    #     ('Ground GPP', ['DL_GPP_RLM', 'DL_GPP_no_physical_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM'],
    #      'GPP with Meteorological Data'),
    #     ('Ground RECO', ['DL_RECO_RLM', 'DL_RECO_no_physical_RLM', 'XGB_RECO_RLM', 'RF_RECO_RLM'],
    #      'RECO with Meteorological Data'),
    #     ('Ground NEE', ['DL_NEE', 'DL_NEE_no_physical', 'XGB_NEE', 'RF_NEE'], 'NEE without Meteorological Data'),
    #     ('Ground GPP', ['DL_GPP', 'DL_GPP_no_physical', 'XGB_GPP', 'RF_GPP'], 'GPP without Meteorological Data'),
    #     ('Ground RECO', ['DL_RECO', 'DL_RECO_no_physical', 'XGB_RECO', 'RF_RECO'], 'RECO without Meteorological Data')
    # ]
    #
    # # 绘制每个组
    # for ax, (ground_col, model_cols, title) in zip(axs.flatten(), groups):
    #     plot_model_biases(ax, ground_col, model_cols, title)
    #
    # # 调整布局
    # plt.tight_layout()
    # plt.show()

#     # Convert to DataFrame
#     df_ground_truth = pd.DataFrame(ground_truth_data)
#     df_predictions = pd.DataFrame(pred_data)
#     biases_with_met = pd.DataFrame({
#         'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
#             'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
#             'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
#             'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
#
#             'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
#             'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
#             'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
#             'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
#
#             'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
#             'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
#             'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
#             'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
#     })
#
#     biases_without_met = pd.DataFrame({
#         'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
#             'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
#             'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
#             'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],
#
#             'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
#             'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
#             'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
#             'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],
#
#             'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
#             'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
#             'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
#             'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
#     })
#
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#
#     # 假设这些 DataFrame 已定义和填充数据
#     # df_ground_truth, biases_with_met, biases_without_met 假定已经存在。
#
#     # 计算均值的函数
#     def calculate_mean_in_bins(df, ground_col, model_cols, num_bins=15):
#         bin_edges = np.linspace(df[ground_col].min(), df[ground_col].max(), num_bins + 1)
#         mean_values = {model: [] for model in model_cols}
#
#         for model in model_cols:
#             binned_df = pd.DataFrame({
#                 'true_value': df[ground_col],
#                 'model_value': df[model]
#             })
#
#             # 计算每个区间的均值
#             for i in range(len(bin_edges) - 1):
#                 mask = (binned_df['true_value'] >= bin_edges[i]) & (binned_df['true_value'] < bin_edges[i + 1])
#                 if mask.any():
#                     mean_value = binned_df['model_value'][mask].mean()
#                 else:
#                     mean_value = np.nan
#                 mean_values[model].append(mean_value)
#
#         return mean_values, bin_edges
#
#
#     # 创建 DataFrame
#     data_with_met = {
#         'Ground NEE': df_ground_truth['Ground_NEE_0'],
#         'DL_NEE_RLM': biases_with_met['DL_NEE_RLM'],
#         'DL_NEE_no_physical_RLM': biases_with_met['DL_NEE_no_physical_RLM'],
#         'XGB_NEE_RLM': biases_with_met['XGB_NEE_RLM'],
#         'RF_NEE_RLM': biases_with_met['RF_NEE_RLM'],
#
#         'Ground GPP': df_ground_truth['Ground_GPP_0'],
#         'DL_GPP_RLM': biases_with_met['DL_GPP_RLM'],
#         'DL_GPP_no_physical_RLM': biases_with_met['DL_GPP_no_physical_RLM'],
#         'XGB_GPP_RLM': biases_with_met['XGB_GPP_RLM'],
#         'RF_GPP_RLM': biases_with_met['RF_GPP_RLM'],
#
#         'Ground RECO': df_ground_truth['Ground_RECO_0'],
#         'DL_RECO_RLM': biases_with_met['DL_RECO_RLM'],
#         'DL_RECO_no_physical_RLM': biases_with_met['DL_RECO_no_physical_RLM'],
#         'XGB_RECO_RLM': biases_with_met['XGB_RECO_RLM'],
#         'RF_RECO_RLM': biases_with_met['RF_RECO_RLM'],
#     }
#
#     data_without_met = {
#         'Ground NEE': df_ground_truth['Ground_NEE_0'],
#         'DL_NEE': biases_without_met['DL_NEE'],
#         'DL_NEE_no_physical': biases_without_met['DL_NEE_no_physical'],
#         'XGB_NEE': biases_without_met['XGB_NEE'],
#         'RF_NEE': biases_without_met['RF_NEE'],
#
#         'Ground GPP': df_ground_truth['Ground_GPP_0'],
#         'DL_GPP': biases_without_met['DL_GPP'],
#         'DL_GPP_no_physical': biases_without_met['DL_GPP_no_physical'],
#         'XGB_GPP': biases_without_met['XGB_GPP'],
#         'RF_GPP': biases_without_met['RF_GPP'],
#
#         'Ground RECO': df_ground_truth['Ground_RECO_0'],
#         'DL_RECO': biases_without_met['DL_RECO'],
#         'DL_RECO_no_physical': biases_without_met['DL_RECO_no_physical'],
#         'XGB_RECO': biases_without_met['XGB_RECO'],
#         'RF_RECO': biases_without_met['RF_RECO'],
#     }
#
#     # 创建带气象数据的 DataFrame
#     df_with_met = pd.DataFrame(data_with_met)
#     df_without_met = pd.DataFrame(data_without_met)
#
#     # 计算带气象数据的均值
#     model_cols_ne_with_met = ['DL_NEE_RLM', 'DL_NEE_no_physical_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM']
#     mean_with_ne, bin_edges_ne = calculate_mean_in_bins(df_with_met, 'Ground NEE', model_cols_ne_with_met)
#
#     model_cols_gpp_with_met = ['DL_GPP_RLM', 'DL_GPP_no_physical_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM']
#     mean_with_gpp, bin_edges_gpp = calculate_mean_in_bins(df_with_met, 'Ground GPP', model_cols_gpp_with_met)
#
#     model_cols_reco_with_met = ['DL_RECO_RLM', 'DL_RECO_no_physical_RLM', 'XGB_RECO_RLM', 'RF_RECO_RLM']
#     mean_with_reco, bin_edges_reco = calculate_mean_in_bins(df_with_met, 'Ground RECO', model_cols_reco_with_met)
#
#     # 计算不带气象数据的均值
#     model_cols_ne_without_met = ['DL_NEE', 'DL_NEE_no_physical', 'XGB_NEE', 'RF_NEE']
#     mean_without_ne, _ = calculate_mean_in_bins(df_without_met, 'Ground NEE', model_cols_ne_without_met)
#
#     model_cols_gpp_without_met = ['DL_GPP', 'DL_GPP_no_physical', 'XGB_GPP', 'RF_GPP']
#     mean_without_gpp, _ = calculate_mean_in_bins(df_without_met, 'Ground GPP', model_cols_gpp_without_met)
#
#     model_cols_reco_without_met = ['DL_RECO', 'DL_RECO_no_physical', 'XGB_RECO', 'RF_RECO']
#     mean_without_reco, _ = calculate_mean_in_bins(df_without_met, 'Ground RECO', model_cols_reco_without_met)
#
#
#     # 可视化结果
#     def plot_biases(mean_with_ne, mean_with_gpp, mean_with_reco, mean_without_ne, mean_without_gpp, mean_without_reco,
#                     bin_edges_ne, bin_edges_gpp, bin_edges_reco):
#         fig, axs = plt.subplots(2, 3, figsize=(18, 10))
#
#         # 第一行：带气象数据的图
#         axs[0, 0].set_title('NEE with Meteorological Data')
#         axs[0, 1].set_title('GPP with Meteorological Data')
#         axs[0, 2].set_title('RECO with Meteorological Data')
#
#         # NEE可视化
#         for model, mean in mean_with_ne.items():
#             bin_centers = (bin_edges_ne[:-1] + bin_edges_ne[1:]) / 2
#             axs[0, 0].plot(bin_centers, mean, marker='o', label=model)
#         axs[0, 0].set_ylabel('Bias with meteorological data')
#         axs[0, 0].grid()
#         axs[0, 0].legend()
#
#         # GPP可视化
#         for model, mean in mean_with_gpp.items():
#             bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
#             axs[0, 1].plot(bin_centers, mean, marker='o', label=model)
#         axs[0, 1].grid()
#
#         # RECO可视化
#         for model, mean in mean_with_reco.items():
#             bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
#             axs[0, 2].plot(bin_centers, mean, marker='o', label=model)
#         axs[0, 2].grid()
#
#         # 设置坐标轴
#         for ax in axs[0, :]:
#             ax.set_xlabel('Ground Value Bins')
#
#         # 第二行：不带气象数据的图
#         axs[1, 0].set_title('NEE without Meteorological Data')
#         axs[1, 1].set_title('GPP without Meteorological Data')
#         axs[1, 2].set_title('RECO without Meteorological Data')
#
#         # NEE可视化
#         for model, mean in mean_without_ne.items():
#             bin_centers = (bin_edges_ne[:-1] + bin_edges_ne[1:]) / 2
#             axs[1, 0].plot(bin_centers, mean, marker='o', label=model)
#         axs[1, 0].set_ylabel('Bias without meteorological data')
#         axs[1, 0].grid()
#         axs[1, 0].legend()
#
#         # GPP可视化
#         for model, mean in mean_without_gpp.items():
#             bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
#             axs[1, 1].plot(bin_centers, mean, marker='o', label=model)
#         axs[1, 1].grid()
#
#         # RECO可视化
#         for model, mean in mean_without_reco.items():
#             bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
#             axs[1, 2].plot(bin_centers, mean, marker='o', label=model)
#         axs[1, 2].grid()
#
#         # 设置坐标轴
#         for ax in axs[1, :]:
#             ax.set_xlabel('Ground Value Bins')
#
#         plt.tight_layout()
#         plt.show()
#
#
#     # 调用可视化函数
#     plot_biases(mean_with_ne, mean_with_gpp, mean_with_reco, mean_without_ne, mean_without_gpp, mean_without_reco,
#                 bin_edges_ne, bin_edges_gpp, bin_edges_reco)
#     #
    # biases_with_met = pd.DataFrame({
    #     'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #
    #         'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #
    #         'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    # })
    #
    # biases_without_met = pd.DataFrame({
    #     'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
    #         'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #         'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #
    #         'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
    #         'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #         'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #
    #         'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
    #         'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
    #         'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    # })
    #
    #
    # # 计算每组模型的均值
    # def calculate_binned_means(df_ground_truth, biases_with_met, biases_without_met, ground_col, num_bins=15):
    #     # 获取ground值
    #     ground_values = df_ground_truth[ground_col]
    #
    #     # bin的区间
    #     bin_edges = np.linspace(ground_values.min(), ground_values.max(), num_bins + 1)
    #
    #     means_with_met = {}
    #     means_without_met = {}
    #
    #     # 计算带气象数据模型的均值
    #     for model in biases_with_met.columns:
    #         binned_means = []
    #         for i in range(len(bin_edges) - 1):
    #             mask = (ground_values >= bin_edges[i]) & (ground_values < bin_edges[i + 1])
    #             if mask.any():
    #                 mean_value = biases_with_met[model][mask].mean()
    #             else:
    #                 mean_value = np.nan
    #             binned_means.append(mean_value)
    #         means_with_met[model] = binned_means
    #
    #     # 计算不带气象数据模型的均值
    #     for model in biases_without_met.columns:
    #         binned_means = []
    #         for i in range(len(bin_edges) - 1):
    #             mask = (ground_values >= bin_edges[i]) & (ground_values < bin_edges[i + 1])
    #             if mask.any():
    #                 mean_value = biases_without_met[model][mask].mean()
    #             else:
    #                 mean_value = np.nan
    #             binned_means.append(mean_value)
    #         means_without_met[model] = binned_means
    #
    #     return means_with_met, means_without_met, bin_edges
    #
    #
    # # 计算均值
    # means_with_met_ne, means_without_met_ne, bin_edges_ne = calculate_binned_means(df_ground_truth, biases_with_met,
    #                                                                                biases_without_met, 'Ground_NEE_0')
    # means_with_met_gpp, means_without_met_gpp, bin_edges_gpp = calculate_binned_means(df_ground_truth, biases_with_met,
    #                                                                                   biases_without_met,
    #                                                                                   'Ground_GPP_0')
    # means_with_met_reco, means_without_met_reco, bin_edges_reco = calculate_binned_means(df_ground_truth,
    #                                                                                      biases_with_met,
    #                                                                                      biases_without_met,
    #                                                                                      'Ground_RECO_0')
    #
    #
    # # 绘制可视化
    # def plot_biases(means_with_met, means_without_met, bin_edges):
    #     fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #
    #     # 第一行：带气象数据
    #     axs[0, 0].set_title('NEE')
    #     axs[0, 1].set_title('GPP')
    #     axs[0, 2].set_title('RECO')
    #
    #     for model in means_with_met.keys():
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         axs[0, 0].plot(bin_centers, means_with_met[model], marker='o', label=model)
    #     axs[0, 0].set_ylabel('Bias with meteorological data')
    #     axs[0, 0].grid()
    #     axs[0, 0].legend()
    #
    #     for model in means_with_met_gpp.keys():
    #         bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
    #         axs[0, 1].plot(bin_centers, means_with_met_gpp[model], marker='o', label=model)
    #     axs[0, 1].grid()
    #
    #     for model in means_with_met_reco.keys():
    #         bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
    #         axs[0, 2].plot(bin_centers, means_with_met_reco[model], marker='o', label=model)
    #     axs[0, 2].grid()
    #
    #     for ax in axs[0, :]:
    #         ax.set_xlabel('True Value Bins')
    #
    #     # 第二行：不带气象数据
    #     # NEE可视化
    #     axs[1, 0].set_title('NEE')
    #     for model in means_without_met.keys():
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #         axs[1, 0].plot(bin_centers, means_without_met[model], marker='o', label=model)
    #     axs[1, 0].set_ylabel('Bias without meteorological data')
    #     axs[1, 0].grid()
    #     axs[1, 0].legend()
    #
    #     for model in means_without_met_gpp.keys():
    #         bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
    #         axs[1, 1].plot(bin_centers, means_without_met_gpp[model], marker='o', label=model)
    #     axs[1, 1].grid()
    #
    #     for model in means_without_met_reco.keys():
    #         bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
    #         axs[1, 2].plot(bin_centers, means_without_met_reco[model], marker='o', label=model)
    #     axs[1, 2].grid()
    #
    #     for ax in axs[1, :]:
    #         ax.set_xlabel('True Value Bins')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # # 调用可视化函数
    # plot_biases(means_with_met_ne, means_without_met_ne, bin_edges_ne)
    # plot_biases(means_with_met_gpp, means_without_met_gpp, bin_edges_gpp)
    # plot_biases(means_with_met_reco, means_without_met_reco, bin_edges_reco)
    # # # 计算偏差
    # def compute_biases():
    #     # 带有气象数据的偏差
    #     biases_with_met = {
    #         'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #         'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
    #
    #         'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #         'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
    #
    #         'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    #         'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0']
    #     }
    #
    #     # 没有气象数据的偏差
    #     biases_without_met = {
    #         'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
    #         'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #         'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],
    #
    #         'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
    #         'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #         'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],
    #
    #         'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
    #         'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
    #         'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    #     }
    #
    #     return biases_with_met, biases_without_met
    #
    #
    # biases_with_met, biases_without_met = compute_biases()
    #
    #
    # # 构建合并 DataFrame
    # def build_biases_dataframe():
    #     rows = {
    #         'Ground NEE': df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE_RLM': biases_with_met['DL_NEE_RLM'],
    #         'DL_NEE_no_physical_RLM': biases_with_met['DL_NEE_no_physical_RLM'],
    #         'XGB_NEE_RLM': biases_with_met['XGB_NEE_RLM'],
    #         'RF_NEE_RLM': biases_with_met['RF_NEE_RLM'],
    #
    #         'Ground GPP': df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP_RLM': biases_with_met['DL_GPP_RLM'],
    #         'DL_GPP_no_physical_RLM': biases_with_met['DL_GPP_no_physical_RLM'],
    #         'XGB_GPP_RLM': biases_with_met['XGB_GPP_RLM'],
    #         'RF_GPP_RLM': biases_with_met['RF_GPP_RLM'],
    #
    #         'Ground RECO': df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO_RLM': biases_with_met['DL_RECO_RLM'],
    #         'DL_RECO_no_physical_RLM': biases_with_met['DL_RECO_no_physical_RLM'],
    #         'XGB_RECO_RLM': biases_with_met['XGB_RECO_RLM'],
    #         'RF_RECO_RLM': biases_with_met['RF_RECO_RLM'],
    #
    #         'Ground NEE': df_ground_truth['Ground_NEE_0'],
    #         'DL_NEE': biases_without_met['DL_NEE'],
    #         'DL_NEE_no_physical': biases_without_met['DL_NEE_no_physical'],
    #         'XGB_NEE': biases_without_met['XGB_NEE'],
    #         'RF_NEE': biases_without_met['RF_NEE'],
    #
    #         'Ground GPP': df_ground_truth['Ground_GPP_0'],
    #         'DL_GPP': biases_without_met['DL_GPP'],
    #         'DL_GPP_no_physical': biases_without_met['DL_GPP_no_physical'],
    #         'XGB_GPP': biases_without_met['XGB_GPP'],
    #         'RF_GPP': biases_without_met['RF_GPP'],
    #
    #         'Ground RECO': df_ground_truth['Ground_RECO_0'],
    #         'DL_RECO': biases_without_met['DL_RECO'],
    #         'DL_RECO_no_physical': biases_without_met['DL_RECO_no_physical'],
    #         'XGB_RECO': biases_without_met['XGB_RECO'],
    #         'RF_RECO': biases_without_met['RF_RECO']
    #     }
    #
    #     return pd.DataFrame(rows)
    #
    #
    # # 生成 DataFrame
    # biases_df = build_biases_dataframe()
    #
    # # 保存为 CSV 文件
    # biases_df.to_csv('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/merged_biases.csv', index=False)
    # # # 计算偏差均值函数
    # # 计算不同区间的模型均值偏差
    # def calculate_mean_bias_in_bins(df, ground_col, model_cols, num_bins=5):
    #     bin_edges = np.linspace(df[ground_col].min(), df[ground_col].max(), num_bins + 1)
    #     binned_means = {model: [] for model in model_cols}
    #
    #     for model in model_cols:
    #         combined = pd.DataFrame({
    #             'true_value': df[ground_col],
    #             'bias': df[model] - df[ground_col]
    #         }).sort_values(by='true_value')
    #
    #         # 遍历每个区间计算均值
    #         for i in range(len(bin_edges) - 1):
    #             mask = (combined['true_value'] >= bin_edges[i]) & (combined['true_value'] < bin_edges[i + 1])
    #             if mask.any():
    #                 mean_value = combined['bias'][mask].mean()
    #             else:
    #                 mean_value = np.nan
    #             binned_means[model].append(mean_value)
    #
    #     return binned_means, bin_edges
    #
    #
    # # 计算偏差均值
    # num_bins = 5
    #
    # # NEE
    # model_cols_ne = ['DL_NEE_RLM', 'DL_NEE_no_physical_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM']
    # mean_bias_ne, bin_edges_ne = calculate_mean_bias_in_bins(biases_df, 'Ground NEE', model_cols_ne, num_bins)
    #
    # # GPP
    # model_cols_gpp = ['DL_GPP_RLM', 'DL_GPP_no_physical_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM']
    # mean_bias_gpp, bin_edges_gpp = calculate_mean_bias_in_bins(biases_df, 'Ground GPP', model_cols_gpp, num_bins)
    #
    # # RECO
    # model_cols_reco = ['DL_RECO_RLM', 'DL_RECO_no_physical_RLM', 'XGB_RECO_RLM', 'RF_RECO_RLM']
    # mean_bias_reco, bin_edges_reco = calculate_mean_bias_in_bins(biases_df, 'Ground RECO', model_cols_reco, num_bins)
    #
    #
    # # 可视化结果
    # def plot_biases(mean_bias_ne, mean_bias_gpp, mean_bias_reco, bin_edges_ne, bin_edges_gpp, bin_edges_reco):
    #     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    #
    #     # 第一行：带气象数据
    #     axs[0, 0].set_title('NEE')
    #     axs[0, 1].set_title('GPP')
    #     axs[0, 2].set_title('RECO')
    #
    #     # NEE可视化
    #     for model, bias in mean_bias_ne.items():
    #         bin_centers = (bin_edges_ne[:-1] + bin_edges_ne[1:]) / 2
    #         axs[0, 0].plot(bin_centers, bias, marker='o', label=model)
    #     axs[0, 0].set_ylabel('Bias with meteorological data')
    #     axs[0, 0].grid()
    #     axs[0, 0].legend()
    #
    #     # GPP可视化
    #     for model, bias in mean_bias_gpp.items():
    #         bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
    #         axs[0, 1].plot(bin_centers, bias, marker='o', label=model)
    #     axs[0, 1].grid()
    #
    #     # RECO可视化
    #     for model, bias in mean_bias_reco.items():
    #         bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
    #         axs[0, 2].plot(bin_centers, bias, marker='o', label=model)
    #     axs[0, 2].grid()
    #
    #     # 设置坐标轴
    #     for ax in axs[0, :]:
    #         ax.set_xlabel('True Value Bins')
    #
    #     # 第二行：不带气象数据
    #     # 重新计算不带气象数据的平均偏差
    #     def calculate_mean_bias_without_met(df, ground_col, model_cols, bin_edges, num_bins=5):
    #         binned_means = {model: [] for model in model_cols}
    #
    #         for model in model_cols:
    #             combined = pd.DataFrame({
    #                 'true_value': df[ground_col],
    #                 'bias': df[model]  # 应该为没有偏差计算的值
    #             }).sort_values(by='true_value')
    #
    #             # 遍历每个区间计算均值
    #             for i in range(len(bin_edges) - 1):
    #                 mask = (combined['true_value'] >= bin_edges[i]) & (combined['true_value'] < bin_edges[i + 1])
    #                 if mask.any():
    #                     mean_value = combined['bias'][mask].mean()
    #                 else:
    #                     mean_value = np.nan
    #                 binned_means[model].append(mean_value)
    #
    #         return binned_means
    #
    #     # 不带气象数据模型列（根据实际情况调整）
    #     mean_bias_ne_no_met = calculate_mean_bias_without_met(biases_df, 'Ground NEE',
    #                                                           ['DL_NEE_no_physical_RLM', 'XGB_NEE_RLM', 'RF_NEE_RLM'],
    #                                                           bin_edges_ne, num_bins)
    #     mean_bias_gpp_no_met = calculate_mean_bias_without_met(biases_df, 'Ground GPP',
    #                                                            ['DL_GPP_no_physical_RLM', 'XGB_GPP_RLM', 'RF_GPP_RLM'],
    #                                                            bin_edges_gpp, num_bins)
    #     mean_bias_reco_no_met = calculate_mean_bias_without_met(biases_df, 'Ground RECO',
    #                                                             ['DL_RECO_no_physical_RLM', 'XGB_RECO_RLM',
    #                                                              'RF_RECO_RLM'], bin_edges_reco, num_bins)
    #
    #     # NEE可视化
    #     axs[1, 0].set_title('NEE')
    #     for model, bias in mean_bias_ne_no_met.items():
    #         bin_centers = (bin_edges_ne[:-1] + bin_edges_ne[1:]) / 2
    #         axs[1, 0].plot(bin_centers, bias, marker='o', label=model)
    #     axs[1, 0].set_ylabel('Bias without meteorological data')
    #     axs[1, 0].grid()
    #     axs[1, 0].legend()
    #
    #     # GPP可视化
    #     axs[1, 1].set_title('GPP')
    #     for model, bias in mean_bias_gpp_no_met.items():
    #         bin_centers = (bin_edges_gpp[:-1] + bin_edges_gpp[1:]) / 2
    #         axs[1, 1].plot(bin_centers, bias, marker='o', label=model)
    #     axs[1, 1].grid()
    #
    #     # RECO可视化
    #     axs[1, 2].set_title('RECO')
    #     for model, bias in mean_bias_reco_no_met.items():
    #         bin_centers = (bin_edges_reco[:-1] + bin_edges_reco[1:]) / 2
    #         axs[1, 2].plot(bin_centers, bias, marker='o', label=model)
    #     axs[1, 2].grid()
    #
    #     # 设置坐标轴
    #     for ax in axs[1, :]:
    #         ax.set_xlabel('True Value Bins')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # # 调用可视化函数
    # plot_biases(mean_bias_ne, mean_bias_gpp, mean_bias_reco, bin_edges_ne, bin_edges_gpp, bin_edges_reco)
    # # def save_biases_to_csv(mean_bias_with_met, mean_bias_without_met, bin_edges_with_met, bin_edges_without_met,
    # #                        filename='E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/biases.csv', num_bins=5):
    # #     # 创建一个空的 DataFrame
    # #     all_biases = []
    # #
    # #     # 处理带有气象的数据模型
    # #     for model_key in mean_bias_with_met.keys():
    # #         for model_index, model_mean in enumerate(mean_bias_with_met[model_key]):
    # #             bin_centers = (bin_edges_with_met[model_key][:-1] + bin_edges_with_met[model_key][1:]) / 2
    # #             data = {
    # #                 'Model': f'{model_key} With Met Model {model_index + 1}',
    # #                 'Bin Center': bin_centers,
    # #                 'Mean Bias': model_mean
    # #             }
    # #             all_biases.append(data)
    # #
    # #     # 处理没有气象的数据模型
    # #     for model_key in mean_bias_without_met.keys():
    # #         for model_index, model_mean in enumerate(mean_bias_without_met[model_key]):
    # #             bin_centers = (bin_edges_without_met[model_key][:-1] + bin_edges_without_met[model_key][1:]) / 2
    # #             data = {
    # #                 'Model': f'{model_key} Without Met Model {model_index + 1}',
    # #                 'Bin Center': bin_centers,
    # #                 'Mean Bias': model_mean
    # #             }
    # #             all_biases.append(data)
    # #
    # #     # 转换为 DataFrame
    # #     biases_df = pd.DataFrame(all_biases)
    # #
    # #     # 将 DataFrame 保存为 CSV 文件
    # #     biases_df.to_csv(filename, index=False)
    # #
    # #
    # # # 调用保存函数
    # # save_biases_to_csv(mean_bias_with_met, mean_bias_without_met, bin_edges_with_met, bin_edges_without_met)
    # import numpy as np
    # import pandas as pd
    #
    #
    # def save_biases_and_raw_to_csv(mean_bias_with_met, mean_bias_without_met, bin_edges_with_met, bin_edges_without_met,
    #                                ground_truth_data, filename='E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/biases_and_raw.csv'):
    #     # 创建一个空的列表以存储最终结果
    #     all_data = []
    #
    #     # 定义处理原始数据和偏差的函数
    #     def append_data(models_biases, met_indicator):
    #         for model_key in models_biases.keys():
    #             for model_index, model_mean in enumerate(models_biases[model_key]):
    #                 bin_centers = (bin_edges_with_met[model_key][:-1] + bin_edges_with_met[model_key][1:]) / 2
    #
    #                 # 确保真值排序
    #                 true_values = np.sort(ground_truth_data[f'Ground_{model_key}_0'])  # 这里用 numpy 的 sort
    #
    #                 # 将均值偏差与真值对应
    #                 for i in range(len(model_mean)):
    #                     all_data.append({
    #                         'Model': f'{model_key} {met_indicator} Model {model_index + 1}',
    #                         'Bin Center': bin_centers[i],
    #                         'Mean Bias': model_mean[i],
    #                         'True Value': true_values[i],  # 对应的真值
    #                         'Bias': true_values[i] - model_mean[i]  # 计算偏差
    #                     })
    #
    #     # 保存带有气象数据的偏差
    #     append_data(mean_bias_with_met, 'With Met')
    #
    #     # 保存没有气象数据的偏差
    #     append_data(mean_bias_without_met, 'Without Met')
    #
    #     # 转换为 DataFrame
    #     biases_and_raw_df = pd.DataFrame(all_data)
    #
    #     # 将 DataFrame 保存为 CSV 文件
    #     biases_and_raw_df.to_csv(filename, index=False)
    #
    #
    # # 调用保存函数
    # save_biases_and_raw_to_csv(mean_bias_with_met, mean_bias_without_met, bin_edges_with_met, bin_edges_without_met,
    #                            ground_truth_data)

    # 使用绝对值 bias

    # # 假设 df_predictions 和 df_ground_truth 已被正确定义
    # # 例如
    # ground_truth_data = {
    #     'Ground_NEE_0': Ground_NEE_0,
    #     'Ground_GPP_0': Ground_GPP_0,
    #     'Ground_RECO_0': Ground_RECO_0
    # }
    #
    # pred_data = {
    #     'DL_NEE_0_RLM': DL_NEE_0_RLM,
    #     'DL_NEE_0_no_physical_RLM': DL_NEE_0_no_physical_RLM,
    #     'XGB_NEE_0_RLM': XGB_NEE_0_RLM,
    #     'RF_NEE_0_RLM': RF_NEE_0_RLM,
    #     'DL_NEE_0': DL_NEE_0,
    #     'DL_NEE_0_no_physical': DL_NEE_0_no_physical,
    #     'XGB_NEE_0': XGB_NEE_0,
    #     'RF_NEE_0': RF_NEE_0,
    #
    #     'DL_GPP_0_RLM': DL_GPP_0_RLM,
    #     'DL_GPP_0_no_physical_RLM': DL_GPP_0_no_physical_RLM,
    #     'XGB_GPP_0_RLM': XGB_GPP_0_RLM,
    #     'RF_GPP_0_RLM': RF_GPP_0_RLM,
    #     'DL_GPP_0': DL_GPP_0,
    #     'DL_GPP_0_no_physical': DL_GPP_0_no_physical,
    #     'XGB_GPP_0': XGB_GPP_0,
    #     'RF_GPP_0': RF_GPP_0,
    #
    #     'DL_RECO_0_RLM': DL_RECO_0_RLM,
    #     'DL_RECO_0_no_physical_RLM': DL_RECO_0_no_physical_RLM,
    #     'XGB_RECO_0_RLM': XGB_RECO_0_RLM,
    #     'RF_RECO_0_RLM': RF_RECO_0_RLM,
    #     'DL_RECO_0': DL_RECO_0,
    #     'DL_RECO_0_no_physical': DL_RECO_0_no_physical,
    #     'XGB_RECO_0': XGB_RECO_0,
    #     'RF_RECO_0': RF_RECO_0,
    # }
    #
    # # Convert to DataFrame
    # df_ground_truth = pd.DataFrame(ground_truth_data)
    # df_predictions = pd.DataFrame(pred_data)
    #
    #
    # # 计算绝对偏差函数
    # def compute_absolute_biases():
    #     biases_with_met = {
    #         'NEE': [
    #             np.abs(df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'])
    #         ],
    #         'GPP': [
    #             np.abs(df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'])
    #         ],
    #         'RECO': [
    #             np.abs(df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'])
    #         ]
    #     }
    #
    #     biases_without_met = {
    #         'NEE': [
    #             np.abs(df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0']),
    #             np.abs(df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'])
    #         ],
    #         'GPP': [
    #             np.abs(df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0']),
    #             np.abs(df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'])
    #         ],
    #         'RECO': [
    #             np.abs(df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0']),
    #             np.abs(df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0'])
    #         ]
    #     }
    #
    #     return biases_with_met, biases_without_met
    #
    #
    # biases_with_met, biases_without_met = compute_absolute_biases()
    #
    #
    # # 更新的计算均值函数
    # def calculate_mean_bias_in_bins(biases, ground_truth_data, num_bins=5):
    #     mean_bias = {}
    #
    #     for model_key in biases.keys():
    #         true_key = f'Ground_{model_key}_0'  # 根据模型类型动态获取真值的键
    #         mean_bias[model_key] = []
    #
    #         if true_key not in ground_truth_data:
    #             print(f"Warning: {true_key} not found in ground_truth_data")  # 添加警告
    #             continue
    #
    #         ground_truth_values = ground_truth_data[true_key]
    #         all_biases = biases[model_key]
    #
    #         # 遍历每个模型
    #         for model_bias in all_biases:
    #             combined = pd.DataFrame({
    #                 'bias': model_bias,
    #                 'true_value': ground_truth_values
    #             })
    #             combined = combined.sort_values(by='true_value')
    #
    #             # 创建区间并计算均值
    #             bin_edges = np.linspace(combined['true_value'].min(), combined['true_value'].max(), num_bins + 1)
    #             binned_means = []
    #
    #             # 遍历每个区间计算均值
    #             for i in range(len(bin_edges) - 1):
    #                 mask = (combined['true_value'] >= bin_edges[i]) & (combined['true_value'] < bin_edges[i + 1])
    #                 if mask.any():
    #                     mean_value = combined['bias'][mask].mean()
    #                 else:
    #                     mean_value = np.nan  # 当无值时返回NaN
    #                 binned_means.append(mean_value)
    #
    #             mean_bias[model_key].append(binned_means)
    #
    #     return mean_bias
    #
    #
    # # 计算偏差均值
    # mean_bias_with_met = calculate_mean_bias_in_bins(biases_with_met, ground_truth_data)
    # mean_bias_without_met = calculate_mean_bias_in_bins(biases_without_met, ground_truth_data)
    #
    #
    # # 可视化结果
    # def plot_biases(mean_bias_with_met, mean_bias_without_met, num_bins=5):
    #     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    #
    #     titles_with_met = ['NEE Bias with Met', 'GPP Bias with Met', 'RECO Bias with Met']
    #     titles_without_met = ['NEE Bias without Met', 'GPP Bias without Met', 'RECO Bias without Met']
    #
    #     for i, (key, title) in enumerate(zip(mean_bias_with_met.keys(), titles_with_met)):
    #         for model_index, model_mean in enumerate(mean_bias_with_met[key]):
    #             axs[0, i].plot(np.linspace(0, 1, num_bins), model_mean, marker='o', label=f'Model {model_index + 1}')
    #
    #         axs[0, i].set_title(title)
    #         axs[0, i].set_xlabel('True Value Bins')
    #         axs[0, i].set_ylabel('Mean Absolute Bias')
    #         axs[0, i].legend()
    #         axs[0, i].grid()
    #
    #     for i, (key, title) in enumerate(zip(mean_bias_without_met.keys(), titles_without_met)):
    #         for model_index, model_mean in enumerate(mean_bias_without_met[key]):
    #             axs[1, i].plot(np.linspace(0, 1, num_bins), model_mean, marker='o', label=f'Model {model_index + 1}')
    #
    #         axs[1, i].set_title(title)
    #         axs[1, i].set_xlabel('True Value Bins')
    #         axs[1, i].set_ylabel('Mean Absolute Bias')
    #         axs[1, i].legend()
    #         axs[1, i].grid()
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # # 调用可视化函数
    # plot_biases(mean_bias_with_met, mean_bias_without_met)

