import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


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
        combined_mask = np.logical_or.reduce([inter_mask_condition, DL_y_qa_mask, DL_mask])
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


if __name__ == "__main__":
    # download the data
    inputdata_path = './data/sites_results/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RL_predict_3.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RL_true_3.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RL_sites_3.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RL_x_mask_3.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RL_y_qa_3.npy', allow_pickle=True)

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

    #CFR with physical-constraints
    CFR_residuals_0_RLM = abs(DL_NEE_0_RLM+DL_GPP_0_RLM - DL_RECO_0_RLM)
    CFR_residuals_1_RLM = abs(DL_NEE_1_RLM +DL_GPP_1_RLM - DL_RECO_1_RLM)
    CFR_residuals_2_RLM = abs(DL_NEE_2_RLM +DL_GPP_2_RLM - DL_RECO_2_RLM)

    CFR_residuals_0_RLM_nophysical = abs(DL_NEE_0_no_physical_RLM + DL_GPP_0_no_physical_RLM - DL_RECO_0_no_physical_RLM)
    CFR_residuals_1_RLM_nophysical = abs(DL_NEE_1_no_physical_RLM + DL_GPP_1_no_physical_RLM - DL_RECO_1_no_physical_RLM)
    CFR_residuals_2_RLM_nophysical = abs(DL_NEE_2_no_physical_RLM + DL_GPP_2_no_physical_RLM - DL_RECO_2_no_physical_RLM)

    #CFR without physical-constraints
    CFR_residuals_0 = abs(DL_NEE_0 + DL_GPP_0 - DL_RECO_0)
    CFR_residuals_1 = abs(DL_NEE_1 + DL_GPP_1- DL_RECO_1)
    CFR_residuals_2 = abs(DL_NEE_2 + DL_GPP_2 - DL_RECO_2)

    CFR_residuals_0_nophysical = abs(DL_NEE_0_no_physical + DL_GPP_0_no_physical - DL_RECO_0_no_physical)
    CFR_residuals_1_nophysical = abs(DL_NEE_1_no_physical + DL_GPP_1_no_physical - DL_RECO_1_no_physical)
    CFR_residuals_2_nophysical = abs(DL_NEE_2_no_physical + DL_GPP_2_no_physical - DL_RECO_2_no_physical)

    # XGB
    XGB_residuals_0_RLM = abs(XGB_NEE_0_RLM + XGB_GPP_0_RLM - XGB_RECO_0_RLM)
    XGB_residuals_1_RLM = abs(XGB_NEE_1_RLM + XGB_GPP_1_RLM - XGB_RECO_1_RLM)
    XGB_residuals_2_RLM = abs(XGB_NEE_2_RLM + XGB_GPP_2_RLM - XGB_RECO_2_RLM)

    XGB_residuals_0 = abs(XGB_NEE_0 + XGB_GPP_0 - XGB_RECO_0)
    XGB_residuals_1 = abs(XGB_NEE_1 + XGB_GPP_1 - XGB_RECO_1)
    XGB_residuals_2 = abs(XGB_NEE_2 + XGB_GPP_2 - XGB_RECO_2)

    # RF
    RF_residuals_0_RLM = abs(RF_NEE_0_RLM + RF_GPP_0_RLM - RF_RECO_0_RLM)
    RF_residuals_1_RLM = abs(RF_NEE_1_RLM + RF_GPP_1_RLM - RF_RECO_1_RLM)
    RF_residuals_2_RLM = abs(RF_NEE_2_RLM + RF_GPP_2_RLM - RF_RECO_2_RLM)

    RF_residuals_0 = abs(RF_NEE_0 + RF_GPP_0 - RF_RECO_0)
    RF_residuals_1 = abs(RF_NEE_1 + RF_GPP_1 - RF_RECO_1)
    RF_residuals_2 = abs(RF_NEE_2 + RF_GPP_2 - RF_RECO_2)

    residuals_list = [CFR_residuals_0_RLM,CFR_residuals_0_RLM_nophysical,XGB_residuals_0_RLM,RF_residuals_0_RLM,
                      CFR_residuals_1_RLM,CFR_residuals_1_RLM_nophysical,XGB_residuals_1_RLM,RF_residuals_1_RLM,
                      CFR_residuals_2_RLM,CFR_residuals_2_RLM_nophysical,XGB_residuals_2_RLM,RF_residuals_2_RLM,
                      CFR_residuals_0, CFR_residuals_0_nophysical, XGB_residuals_0, RF_residuals_0,
                      CFR_residuals_1, CFR_residuals_1_nophysical, XGB_residuals_1, RF_residuals_1,
                      CFR_residuals_2, CFR_residuals_2_nophysical, XGB_residuals_2, RF_residuals_2,
                      ]
    # DL_residuals = DL_NEE_2 - (DL_RECO_2 - DL_GPP_2)
    # abs_DL_residuals = abs(DL_residuals)
    # 为每个变量计算最小值、最大值、中位数、均值和标准差
    for residuals in residuals_list:
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        median_val = np.median(residuals)
        mean_val = np.mean(residuals)
        std_dev = np.std(residuals)
        print(f"{min_val:.2f},{max_val:.2f},{median_val:.2f},{mean_val:.2f},{std_dev:.2f}")
    # # Create the scatter plot with density-based color mapping
    # # # **********************************
    # # # 一张大图可视化四个子图
    # # 创建一个包含多个子图的大图
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # Define the variables to be processed along with their corresponding data
    # variables = ['_NEE_residuals', '_GPP_residuals']
    # RF_variables = ['abs_Ground_residuals', 'abs_Ground_residuals']
    # DL_variables = ['abs_DL_residuals', 'abs_RF_residuals']
    #
    # # Define the color map
    # color_map = ['Blues', 'Oranges', 'Greens', 'Reds']
    #
    # for i in range(len(variables)):
    #     # Get the current variable and its corresponding RF and DL variables
    #     variable = variables[i]
    #     RF_variable = RF_variables[i]
    #     DL_variable = DL_variables[i]
    #
    #     # Get the data for the current variable
    #     x = globals()[RF_variable]  # Get the data for the RF variable
    #     y = globals()[DL_variable]  # Get the data for the DL variable
    #
    #     # Calculate point density
    #     xy = np.vstack([x, y]).astype(float)
    #     z = gaussian_kde(xy)(xy)
    #     idx = z.argsort()
    #     x, y, z = x[idx], y[idx], z[idx]
    #
    #     # Normalize the density values to the range 0 to 1
    #     z = (z - np.min(z)) / (np.max(z) - np.min(z))
    #
    #     # Plot the scatter density plot and set the frequency (probability)
    #     scatter = axs[i].scatter(x, y, c=z, cmap='Spectral_r', s=20)
    #     plt.colorbar(scatter, ax=axs[i], label='Density')
    #     min_val = min(np.min(y), np.min(y))
    #     max_val = max(np.max(y), np.max(y))
    #     axs[i].set_xlim([min_val, max_val])
    #     axs[i].set_ylim([min_val, max_val])
    #     axs[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    #
    #     # Calculate RMSE, R2, and bias evaluation metrics
    #     rmse = np.sqrt(mean_squared_error(x, y))
    #     r2 = r2_score(x, y)
    #     n = len(x)
    #     bias = np.mean(x - y)
    #
    #     # Add text annotations
    #     text = "RMSE = {:.2f}\nR2 = {:.2f}\nn = {}".format(rmse, r2, n)
    #     axs[i].text(0.05, 0.95, text, verticalalignment='top', transform=axs[i].transAxes,
    #                 bbox=dict(facecolor='none', edgecolor='red'))
    #
    #     # Set axis labels and title
    #     axs[i].set_xlabel(RF_variable)
    #     axs[i].set_ylabel(DL_variable)
    #     axs[i].set_title( DL_variable + " vs. " + RF_variable)
    #
    # # Add a main title to the entire plot
    # plt.suptitle(
    #     'Comparison of absolute value residuals of predicted values and absolute value residuals of observed values')
    #
    # # Display the plot
    # plt.tight_layout()
    # plt.show()