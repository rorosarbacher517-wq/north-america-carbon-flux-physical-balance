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


## real,pred = y_test_plot_i, y_pred_plot_i
def get_regression_line(real, pred, data_range=(0, 110)):
    # 拟合（若换MK，自行操作）最小二乘
    def slope(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

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

    # Convert to DataFrame
    df_ground_truth = pd.DataFrame(ground_truth_data)
    df_predictions = pd.DataFrame(pred_data)

    # # Calculate biases
    NEE_bias_with_met = [
        df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0']
    ]

    NEE_bias_without_met = [
        df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
        df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0']
    ]

    GPP_bias_with_met = [
        df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0']
    ]

    GPP_bias_without_met = [
        df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
        df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0']
    ]

    RECO_bias_with_met = [
        df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0']
    ]

    RECO_bias_without_met = [
        df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
        df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    ]
    # # Calculate biases
    NEE_bias_with_met = [
        df_ground_truth['Ground_NEE_0'],
        df_predictions['DL_NEE_0_RLM'],
        df_predictions['DL_NEE_0_no_physical_RLM'],
        df_predictions['XGB_NEE_0_RLM'],
        df_predictions['RF_NEE_0_RLM']
    ]

    NEE_bias_without_met = [
        df_ground_truth['Ground_NEE_0'],
        df_predictions['DL_NEE_0'],
        df_predictions['DL_NEE_0_no_physical'],
        df_predictions['XGB_NEE_0'],
        df_predictions['RF_NEE_0']
    ]

    GPP_bias_with_met = [
        df_ground_truth['Ground_GPP_0'],
        df_predictions['DL_GPP_0_RLM'],
        df_predictions['DL_GPP_0_no_physical_RLM'],
        df_predictions['XGB_GPP_0_RLM'],
        df_predictions['RF_GPP_0_RLM']
    ]

    GPP_bias_without_met = [
        df_ground_truth['Ground_GPP_0'],
        df_predictions['DL_GPP_0'],
        df_predictions['DL_GPP_0_no_physical'],
        df_predictions['XGB_GPP_0'],
        df_predictions['RF_GPP_0']
    ]

    RECO_bias_with_met = [
        df_ground_truth['Ground_RECO_0'],
        df_predictions['DL_RECO_0_RLM'],
        df_predictions['DL_RECO_0_no_physical_RLM'],
        df_predictions['XGB_RECO_0_RLM'],
        df_predictions['RF_RECO_0_RLM']
    ]

    RECO_bias_without_met = [
        df_ground_truth['Ground_RECO_0'],
        df_predictions['DL_RECO_0'],
        df_predictions['DL_RECO_0_no_physical'],
        df_predictions['XGB_RECO_0'],
        df_predictions['RF_RECO_0']
    ]
#
#     # Create a 2x3 subplot
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#     plt.rcParams['font.family'] = 'Times New Roman'
#
#     # Define titles and x-labels
#     titles = ['NEE with Meteorological Data', 'GPP with Meteorological Data', 'RECO with Meteorological Data',
#               'NEE without Meteorological Data', 'GPP without Meteorological Data', 'RECO without Meteorological Data']
#     x_labels = ['DL_with', 'DL_without', 'XGBoost', 'RF'] # 'Ground',
#
#     # Fill the plots for the first row (with meteorological data)
#     axs[0, 0].boxplot(NEE_bias_with_met, labels=x_labels)
#     axs[0, 0].set_title(titles[0])
#     # axs[0, 0].set_ylabel('Bias (Model estimated - Flux tower)')
#     axs[0, 0].set_ylabel('Flux values')
#     axs[0, 0].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     axs[0, 1].boxplot(GPP_bias_with_met, labels=x_labels)
#     axs[0, 1].set_title(titles[1])
#     axs[0, 1].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     axs[0, 2].boxplot(RECO_bias_with_met, labels=x_labels)
#     axs[0, 2].set_title(titles[2])
#     axs[0, 2].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     # Fill the plots for the second row (without meteorological data)
#     axs[1, 0].boxplot(NEE_bias_without_met, labels=x_labels)
#     axs[1, 0].set_title(titles[3])
#     # axs[1, 0].set_ylabel('Bias (Model estimated - Flux tower)')
#     axs[1, 0].set_ylabel('Flux values')
#     axs[1, 0].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     axs[1, 1].boxplot(GPP_bias_without_met, labels=x_labels)
#     axs[1, 1].set_title(titles[4])
#     axs[1, 1].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     axs[1, 2].boxplot(RECO_bias_without_met, labels=x_labels)
#     axs[1, 2].set_title(titles[5])
#     axs[1, 2].axhline(0, color='gray', linestyle='--')  # Add dashed line at y=0
#
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()
# # #
#     X_variables = ['Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0','Ground_NEE_0','Ground_NEE_0', 'Ground_NEE_0']
#     Y_variables = ['DL_NEE_0_RLM','DL_NEE_0_no_physical_RLM', 'XGB_NEE_0_RLM', 'RF_NEE_0_RLM','DL_NEE_0','DL_NEE_0_no_physical','XGB_NEE_0', 'RF_NEE_0']
#
#     model_subtitle = ['Time series deep learning model\nwith physical constraints','Time series deep learning model\nwithout physical constraints','XGBoost', 'RF'] # 'CFR without physical constraints',
#     # model_subtitle = ['NEE', 'GPP', 'RECO']  # 'CFR without physical constraints',
#     ax_subtitle = ['(a)', '(b)', '(c)', '(d)','(e)','(f)','(g)','(h)'] # ,'(g)','(h)'
#     import matplotlib.pyplot as plt
#     # 设置共享colorbar位置
#     # divider = make_axes_locatable(axes[1, 3])
#     # cax = divider.append_axes("right", size="5%", pad=0.1)
#
#     # 循环绘制子图
#     scatter_handles = []
#     plt.rcParams['font.family'] = 'Times New Roman'
#     fig, axs = plt.subplots(3, 4, figsize=(12,9))
#
#     # Find the overall min and max values for x and y
#     overall_x_min = min(globals()[var].min() for var in X_variables)
#     overall_x_max = max(globals()[var].max() for var in X_variables)
#     overall_y_min = min(globals()[var].min() for var in Y_variables)
#     overall_y_max = max(globals()[var].max() for var in Y_variables)
#
#     # Y-axis limits for the bias boxplots
#     y_lim = [-5, 5]  # Adjust as appropriate for your data
#
#     # Create boxplots in the first row
#     for i in range(4):
#         # Get biases for (i and i + 4)
#         x1 = globals()[X_variables[i + 0]][:1000]
#         y1 = globals()[Y_variables[i + 0]][:1000]
#         bias1 = y1 - x1
#
#         x2 = globals()[X_variables[i + 4]][:1000]
#         y2 = globals()[Y_variables[i + 4]][:1000]
#         bias2 = y2 - x2
#
#         # Use different colors for the box plots
#         box_colors = ['lightblue', 'lightcoral']
#         axs[0, i].boxplot([bias1, bias2], labels=[f'With Met. Data', f'Without Met. Data'],
#                           patch_artist=True, boxprops=dict(facecolor=box_colors[0]),
#                           medianprops=dict(color='black'))
#
#         # axs[0, i].set_title(f'Bias of {ax_subtitle[i]}', fontsize=10)
#         axs[0, i].axhline(y=0, color='grey', linestyle='--')
#         axs[0, i].set_ylim(y_lim)  # Maintain consistent y-axis limits
#
#     # Plot scatter and density on the second and third rows
#     for i in range(4):
#         row = 1  # Middle row
#         col = i
#         x = globals()[X_variables[i]][:1000]
#         y = globals()[Y_variables[i]][:1000]
#
#         # Calculate and display scatter with a density plot
#         xy = np.vstack([x, y]).astype(float)
#         z = gaussian_kde(xy)(xy)
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
#         z = (z - np.min(z)) / (np.max(z) - np.min(z))
#
#         axs[row, col].set_xlim([overall_x_min, overall_x_max])
#         axs[row, col].set_ylim([overall_y_min, overall_y_max])
#         axs[row, col].plot([overall_x_min, overall_x_max], [overall_y_min, overall_y_max], color='#535CA8',
#                            linestyle='--', linewidth=1)
#         scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
#
#         # Calculate and display metrics
#         rmse = np.sqrt(mean_squared_error(x, y))
#         cc = np.corrcoef(x, y)[0, 1]
#         n = len(x)
#         text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}\nn = {n}"
#         axs[row, col].text(0.6, 0.21, text, verticalalignment='top', transform=axs[row, col].transAxes, color='red',
#                            fontweight='bold')
#
#     # Set general y-axis labels for each row
#     axs[0, 0].set_ylabel('Bias (Model estimated - Flux tower)', fontsize=12)
#     axs[1, 0].set_ylabel('Input with meteorological data', fontsize=12)
#     axs[2, 0].set_ylabel('Input without meteorological data', fontsize=12)
#
#     # Set model subtitles for the last row
#     for i in range(4):
#         axs[2, i].text(0.5, -0.2, model_subtitle[i], ha='center', va='center', fontsize=10,
#                        transform=axs[2, i].transAxes)
#
#     plt.tight_layout()
#     plt.show()
#     for i in range(8):
#         row = (i // 4) + 1  # Shift all original plots down by one row
#         col = i % 4
#         x_variable = X_variables[i]
#         y_variable = Y_variables[i]
#         x = globals()[x_variable][:1000]
#         y = globals()[y_variable][:1000]
#
#         bias = y - x  # Calculate bias for boxplot
#
#         if i < 4:
#             axs[0, col].boxplot(bias)
#             axs[0, col].set_title(f'Bias of {ax_subtitle[i]}', fontsize=10)
#             axs[0, col].set_ylabel('Bias (Pred - Obs)')
#             axs[0, col].axhline(y=0, color='grey', linestyle='--')
#
#         # Plot scatter and diagonal line on the middle and bottom rows
#         xy = np.vstack([x, y]).astype(float)
#         z = gaussian_kde(xy)(xy)
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
#         z = (z - np.min(z)) / (np.max(z) - np.min(z))
#
#         axs[row, col].set_xlim([overall_x_min, overall_x_max])
#         axs[row, col].set_ylim([overall_y_min, overall_y_max])
#         axs[row, col].plot([overall_x_min, overall_x_max], [overall_y_min, overall_y_max], color='#535CA8', linestyle='--', linewidth=1)
#         scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
#
#         # Calculate and display metrics
#         rmse = np.sqrt(mean_squared_error(x, y))
#         cc = np.corrcoef(x, y)[0, 1]
#         n = len(x)
#         text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}\nn = {n}"
#         axs[row, col].text(0.6, 0.21, text, verticalalignment='top', transform=axs[row, col].transAxes, color='red', fontweight='bold')
#
#         if row == 1:  # Middle row
#             axs[row, col].set_title(model_subtitle[col], pad=10)
#
#         if row == 1 and col == 0:
#             axs[row, col].set_ylabel('(1) Input with meteorological data')
#         elif row == 2 and col == 0:
#             axs[row, col].set_ylabel('(2) Input without meteorological data')
#
#     plt.tight_layout()
#     plt.show()
#     for i in range(8):  # Modify the range to match the 2x3 grid
#         row = i // 4
#         col = i % 4
#         x_variable = X_variables[i]
#         y_variable = Y_variables[i]
#         x = globals()[x_variable][:1000]
#         y = globals()[y_variable][:1000]
#
#         xy = np.vstack([x, y]).astype(float)
#         z = gaussian_kde(xy)(xy)
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
#
#         z = (z - np.min(z)) / (np.max(z) - np.min(z))
# #
#         # if col == 0:
#         #     min_val_x = min(np.min(Ground_NEE_0_RLM), np.min(Ground_NEE_0_RLM))
#         #     max_val_x = max(np.max(Ground_NEE_0_RLM), np.max(Ground_NEE_0_RLM))
#         #     min_val_y = min(np.min(DL_NEE_0_RLM), np.min(DL_NEE_0_no_physical_RLM))
#         #     max_val_y = max(np.max(DL_NEE_0_RLM), np.max(DL_NEE_0_no_physical_RLM))
#         #     axs[row, col].set_xlim([min_val_x, max_val_x])
#         #     axs[row, col].set_ylim([min_val_y, max_val_y])
#         # elif col == 1:
#         #     min_val_x = min(np.min(Ground_GPP_0_RLM), np.min(Ground_GPP_0_RLM))
#         #     max_val_x = max(np.max(Ground_GPP_0_RLM), np.max(Ground_GPP_0_RLM))
#         #     min_val_y = min(np.min(DL_GPP_0_RLM), np.min(DL_GPP_0_no_physical_RLM))
#         #     max_val_y = max(np.max(DL_GPP_0_RLM), np.max(DL_GPP_0_no_physical_RLM))
#         #     axs[row, col].set_xlim([min_val_x, max_val_x])
#         #     axs[row, col].set_ylim([min_val_y, max_val_y])
#         # elif col == 2:
#         #     min_val_x = min(np.min(Ground_RECO_0_RLM), np.min(Ground_RECO_0_RLM))
#         #     max_val_x = max(np.max(Ground_RECO_0_RLM), np.max(Ground_RECO_0_RLM))
#         #     min_val_y = min(np.min(DL_RECO_0_RLM), np.min(DL_RECO_0_no_physical_RLM))
#         #     max_val_y = max(np.max(DL_RECO_0_RLM), np.max(DL_RECO_0_no_physical_RLM))
#         #     axs[row, col].set_xlim([min_val_x, max_val_x])
#         #     axs[row, col].set_ylim([min_val_y, max_val_y])
#
#         # axs[row, col].plot([min_val_x, max_val_x], [min_val_y, max_val_y], color='#535CA8', linestyle='--', linewidth=1)
#         # 同一行的横纵轴一样
#
#         # min_val = min(np.min(y), np.min(y))
#         # max_val = max(np.max(y), np.max(y))
#         # axs[row, col].set_xlim([min_val, max_val])
#         # axs[row, col].set_ylim([min_val, max_val])
#         # axs[row, col].plot([min_val, max_val], [min_val, max_val],  color='#535CA8',
#         #                    linestyle='--', linewidth=1)
#         # #535CA8
#         # 不同模型单个变量对比
#         axs[row, col].set_xlim([overall_x_min, overall_x_max])
#         axs[row, col].set_ylim([overall_y_min, overall_y_max])
#         axs[row, col].plot([overall_x_min, overall_x_max], [overall_y_min, overall_y_max], color='#535CA8',
#                            linestyle='--', linewidth=1)
#         scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
#         scatter_handles.append(scatter)
#
#         rmse = np.sqrt(mean_squared_error(x, y))
#         r2 = r2_score(x, y)
#         n = len(x)
#         bias = np.mean(x - y)
#         cc = np.corrcoef(x, y)[0, 1]
#         mean_value = np.mean(abs(x))
#         rrmse = rmse / mean_value
#         axs[row, col].text(0.05, 0.95, ax_subtitle[i],verticalalignment='top', transform=axs[row, col].transAxes,
#                            color='black', fontname='Times New Roman')
#         text = "RMSE = {:.2f}\nCC = {:.2f}\nn = {}".format(rmse, cc, n)
#         axs[row, col].text(0.6, 0.21, text, verticalalignment='top', transform=axs[row, col].transAxes, color='red',
#                            fontweight='bold',
#                            fontname='Times New Roman')
#         if row == 0:
#             axs[row, col].set_title(model_subtitle[i], fontname='Times New Roman', pad=10)
#
#         if row == 0 and col == 0:
#             axs[row, col].set_ylabel('(1) Input with meteorological data')
#             # axs[row, col].set_ylabel('CFR with physical constraints')
#         elif row == 1 and col == 0:
#             axs[row, col].set_ylabel('(2) Input without meteorological data')
#             # axs[row, col].set_ylabel('CFR without physical constraints')
#
#         # # Add binned bias plot
#         # if row == 0:  # Only for top row plots
#         #     bin_means = np.linspace(np.min(x), np.max(x), 10)
#         #     bin_centers = (bin_means[:-1] + bin_means[1:]) / 2
#         #     digitized = np.digitize(x, bin_means)
#         #     bias_binned = [np.mean(y[digitized == j] - x[digitized == j]) for j in range(1, len(bin_means))]
#         #
#         #     axs[2, col].plot(bin_centers, bias_binned, marker='o', color='blue')
#         #     axs[2, col].axhline(y=0, color='grey', linestyle='--')
#         #     axs[2, col].set_ylim([-0.2, 0.2])  # Adjust based on your data
#         #     axs[2, col].set_title(f'Bias vs. {x_variable}')
#         #     axs[2, col].set_xlabel('Flux Tower Value')
#         #     axs[2, col].set_ylabel('Bias (Pred - Obs)')
#     # Share x and y axis labels
#     fig.subplots_adjust(right=0.9)
#     position = fig.add_axes([0.92, 0.15, 0.015, .75])  # position [left, bottom, right, top]
#     cb = fig.colorbar(scatter_handles[0], cax=position)
#     # Set colorbar label font, size, and family
#     colorbarfontdict = {"size": 12, "color": "k", 'family': 'Times New Roman'}
#     cb.ax.set_title('Density', fontdict=colorbarfontdict, pad=10)
#     cb.ax.tick_params(labelsize=12, direction='in')
#
#     # Set labels
#     fig.text(0.5, 0.05, 'Flux tower NEE (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center', fontsize=12,
#              fontname='Times New Roman')
#     # fig.text(0.03, 0.5, 'Prediction of CFR using RLF as input (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center', rotation='vertical',
#     #          fontsize=12, fontname='Times New Roman')
#     fig.text(0.04, 0.55, 'Model estimated NEE (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center',
#              rotation='vertical',
#              fontsize=12, fontname='Times New Roman')
#     # fig.text(0.068, 0.7, '(1) Using RLFM as input', ha='center', va='center', rotation='vertical', fontsize=12,
#     #          fontname='Times New Roman')
#     # fig.text(0.068, 0.3, '(2) Using RLF as input', ha='center', va='center', rotation='vertical', fontsize=12,
#     #          fontname='Times New Roman')
#
#     # Adjust overall layout
#     plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.2, hspace=0.2)
#     # base_name = 'Comparison of predicted NEE by the CFR model and machine learning models using RLFM(1) or RLF(2) as input'
#     # plt.suptitle(base_name, ha='center', fontsize=12, y=0.99, fontname='Times New Roman')
#
#     resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
#     plt.savefig(resluts_path + 'NEE comparison of all models' + '_scenario 9' + '.png', dpi=300)
#     # Display the plot
#     plt.show()