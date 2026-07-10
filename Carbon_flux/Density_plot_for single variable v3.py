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
    #
    XGB_GPP_predict_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true_RLM = np.load(inputdata_path + 'XGB_RLM_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true_RLM = np.load(inputdata_path + 'XGB_RLM_RECO_true_3.npy', allow_pickle=True)


    # print(min(XGB_GPP_predict),min(XGB_GPP_predict),min(XGB_GPP_predict_RLM),min(XGB_GPP_predict_RLM))
    # print(min(XGB_RECO_predict), min(XGB_RECO_predict), min(XGB_RECO_predict_RLM), min(XGB_RECO_predict_RLM))
#     # 第一组值:插值与不插值的一起
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

    # print(min(Ground_GPP_0_RLM),min(DL_GPP_0_RLM),min(DL_GPP_0_no_physical_RLM),min(XGB_GPP_0),min(RF_GPP_0_RLM))
    X_variables = ['Ground_GPP_0_RLM','Ground_GPP_0_RLM','Ground_GPP_0_RLM','Ground_GPP_0_RLM','Ground_GPP_0','Ground_GPP_0','Ground_GPP_0', 'Ground_GPP_0']
    Y_variables = ['DL_GPP_0_RLM','DL_GPP_0_no_physical_RLM', 'XGB_GPP_0_RLM', 'RF_GPP_0_RLM','DL_GPP_0','DL_GPP_0_no_physical','XGB_GPP_0', 'RF_GPP_0']
    # X_variables = ['Ground_GPP_0_RLM', 'Ground_GPP_0_RLM', 'Ground_RECO_0_RLM', 'Ground_NEE_0', 'Ground_GPP_0', 'Ground_RECO_0']
    # Y_variables = ['DL_NEE_0_no_physical_RLM', 'DL_GPP_0_no_physical_RLM', 'DL_RECO_0_no_physical_RLM', 'DL_NEE_0_no_physical', 'DL_GPP_0_no_physical', 'DL_RECO_0_no_physical']
    # X_variables = ['Ground_NEE_0', 'Ground_GPP_0', 'Ground_RECO_0',
    #                'Ground_NEE_0', 'Ground_GPP_0', 'Ground_RECO_0'] # 'Ground_NEE_0_RLM', 'Ground_GPP_0_RLM', 'Ground_RECO_0_RLM',
    # Y_variables = ['DL_NEE_0', 'DL_GPP_0', 'DL_RECO_0',
    #                'DL_NEE_0_no_physical', 'DL_GPP_0_no_physical', 'DL_RECO_0_no_physical'] # 'DL_NEE_0_no_physical', 'DL_GPP_0_no_physical', 'DL_RECO_0_no_physical'

    # X_variables = ['Ground_RECO_0_RLM', 'Ground_RECO_0_RLM', 'Ground_RECO_0_RLM', 'Ground_RECO_0_RLM', 'Ground_RECO_0', 'Ground_RECO_0', 'Ground_RECO_0', 'Ground_RECO_0']
    # Y_variables = ['DL_RECO_0_RLM', 'DL_RECO_0_no_physical_RLM', 'XGB_RECO_0_RLM', 'RF_RECO_0_RLM', 'DL_RECO_0', 'DL_RECO_0_no_physical', 'XGB_RECO_0', 'RF_RECO_0']
    # X_labels = ['Ground observation','Ground observation','Ground observation','Ground observation','Ground observation','Ground observation','Ground observation','Ground observation']
    # Y_labels = ['CFR NEE overall', 'CFR NEE overall', 'XGB NEE overall', 'RF NEE overall']
    model_subtitle = ['Physics-aware transformer','Transformer', 'XGBoost', 'RF']# 'CFR without physical constraints',
    # model_subtitle = ['NEE', 'GPP', 'RECO']  # 'CFR without physical constraints',
    ax_subtitle = ['(a)', '(b)', '(c)', '(d)','(e)','(f)','(g)','(h)'] # ,'(g)','(h)'
    import matplotlib.pyplot as plt
    # 设置共享colorbar位置
    # divider = make_axes_locatable(axes[1, 3])
    # cax = divider.append_axes("right", size="5%", pad=0.1)

    # 循环绘制子图
    scatter_handles = []
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(2, 4, figsize=(16,9))

    # Find the overall min and max values for x and y
    overall_x_min = min(globals()[var].min() for var in X_variables)
    overall_x_max = max(globals()[var].max() for var in X_variables)
    overall_y_min = min(globals()[var].min() for var in Y_variables)
    overall_y_max = max(globals()[var].max() for var in Y_variables)
    # common_min = min(overall_x_min, overall_y_min)
    common_min = 0
    common_max = max(overall_x_max, overall_y_max)
    # for i in range(8):
    #     row = (i // 4) + 1  # Shift all original plots down by one row
    #     col = i % 4
    #     x_variable = X_variables[i]
    #     y_variable = Y_variables[i]
    #     x = globals()[x_variable][:1000]
    #     y = globals()[y_variable][:1000]
    #
    #     bias = y - x  # Calculate bias for boxplot
    #
    #     if i < 4:
    #         axs[0, col].boxplot(bias)
    #         axs[0, col].set_title(f'Bias of {ax_subtitle[i]}', fontsize=10)
    #         axs[0, col].set_ylabel('Bias (Pred - Obs)')
    #         axs[0, col].axhline(y=0, color='grey', linestyle='--')
    #
    #     # Plot scatter and diagonal line on the middle and bottom rows
    #     xy = np.vstack([x, y]).astype(float)
    #     z = gaussian_kde(xy)(xy)
    #     idx = z.argsort()
    #     x, y, z = x[idx], y[idx], z[idx]
    #     z = (z - np.min(z)) / (np.max(z) - np.min(z))
    #
    #     axs[row, col].set_xlim([overall_x_min, overall_x_max])
    #     axs[row, col].set_ylim([overall_y_min, overall_y_max])
    #     axs[row, col].plot([overall_x_min, overall_x_max], [overall_y_min, overall_y_max], color='#535CA8', linestyle='--', linewidth=1)
    #     scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
    #
    #     # Calculate and display metrics
    #     rmse = np.sqrt(mean_squared_error(x, y))
    #     cc = np.corrcoef(x, y)[0, 1]
    #     n = len(x)
    #     text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}\nn = {n}"
    #     axs[row, col].text(0.6, 0.21, text, verticalalignment='top', transform=axs[row, col].transAxes, color='red', fontweight='bold')
    #
    #     if row == 1:  # Middle row
    #         axs[row, col].set_title(model_subtitle[col], pad=10)
    #
    #     if row == 1 and col == 0:
    #         axs[row, col].set_ylabel('(1) Input with meteorological data')
    #     elif row == 2 and col == 0:
    #         axs[row, col].set_ylabel('(2) Input without meteorological data')
    #
    # plt.tight_layout()
    # plt.show()
    for i in range(8):  # Modify the range to match the 2x3 grid
        row = i // 4
        col = i % 4
        x_variable = X_variables[i]
        y_variable = Y_variables[i]
        x = globals()[x_variable]
        y = globals()[y_variable]

        xy = np.vstack([x, y]).astype(float)
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        z = (z - np.min(z)) / (np.max(z) - np.min(z))

        ## 将较低密度（<0.1）的值替换为 NaN，以便在散点图中显示为白色
        z_cleaned = np.where(z < 0.1, np.nan, z)

        # 设置坐标轴范围
        axs[row, col].set_xlim([common_min, common_max])
        axs[row, col].set_ylim([common_min, common_max])
        # Setting integer ticks
        num_ticks = 5
        ticks_x = np.linspace(common_min, common_max, num_ticks)
        ticks_y = np.linspace(common_min, common_max, num_ticks)
        ticks_x = np.round(ticks_x).astype(int)
        ticks_y = np.round(ticks_y).astype(int)

        axs[row, col].set_xticks(ticks_x)
        axs[row, col].set_yticks(ticks_y)
        # 设置字体大小
        axs[row, col].tick_params(axis='both', labelsize=16)

        # 绘制主对角线
        axs[row, col].plot([common_min, common_max], [common_min, common_max], color='#535CA8', linestyle='--', linewidth=1)

        # 绘制散点图，使用 NaN 处理白色点
        scatter = axs[row, col].scatter(x, y, c=z_cleaned, cmap='Spectral_r', s=20, edgecolor='none')

        # 另绘制低于阈值的点为白色
        low_density_indices = np.where(z < 0.1)[0]
        if low_density_indices.size > 0:
            axs[row, col].scatter(x[low_density_indices], y[low_density_indices], color='white', s=20)

        scatter_handles.append(scatter)


        # axs[row, col].set_xlim([common_min, common_max])
        # axs[row, col].set_ylim([common_min, common_max])
        # # Set font size for tick labels
        # axs[row, col].tick_params(axis='both', labelsize=16)
        # axs[row, col].plot([common_min, common_max], [common_min, common_max], color='#535CA8',linestyle='--', linewidth=1)
        # scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
        # scatter_handles.append(scatter)

        rmse = np.sqrt(mean_squared_error(x, y))
        r2 = r2_score(x, y)
        n = len(x)
        bias = np.mean(x - y)
        cc = np.corrcoef(x, y)[0, 1]
        mean_value = np.mean(abs(x))
        rrmse = rmse / mean_value
        # axs[row, col].text(0.05, 0.95, ax_subtitle[i],verticalalignment='top', transform=axs[row, col].transAxes,
        #                    color='black', fontname='Times New Roman')
        # Display metrics on the plot
        text = "RMSE = {:.2f}\nCC = {:.2f}".format(rmse, cc)
        axs[row, col].text(0.05, 0.97, text, verticalalignment='top', transform=axs[row, col].transAxes, color='red',
                           fontweight='bold', fontname='Times New Roman', fontsize=16)

        # Set axis labels for each subplot
        axs[row, col].set_xlabel('Flux tower', fontname='Times New Roman', fontsize=18)
        axs[row, col].set_ylabel('Model estimated', fontname='Times New Roman', fontsize=18)

        # Set titles for the first row
        if row == 0:
            axs[row, col].set_title(model_subtitle[i], fontname='Times New Roman', pad=10, fontsize=18)

        # Share x and y axis labels
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.92, 0.15, 0.015, .75])  # position [left, bottom, right, top]
    cb = fig.colorbar(scatter_handles[0], cax=position)

    # Set colorbar label font, size, and family
    colorbarfontdict = {"size": 18, "color": "k", 'family': 'Times New Roman'}
    cb.ax.set_title('Density', fontdict=colorbarfontdict, pad=10)
    cb.ax.tick_params(labelsize=18, direction='in')

    # Change color of the colorbar to a lighter shade
    for label in cb.ax.get_yticklabels():
        label.set_color('black')  # Change the color to a lighter shade

    # Set labels for the entire figure
    fig.text(0.5, 0.05, 'GPP (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center', fontsize=18,
             fontname='Times New Roman')
    fig.text(0.03, 0.7, '(1) Input with\nmeteorological data', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Times New Roman')
    fig.text(0.03, 0.3, '(2) Input without\nmeteorological data', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Times New Roman')

    # Adjust overall layout
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.35, hspace=0.35)

    # base_name = 'Comparison of predicted GPP by the CFR model and machine learning models using RLFM(1) or RLF(2) as input'
    # plt.suptitle(base_name, ha='center', fontsize=12, y=0.99, fontname='Times New Roman')

    resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    plt.savefig(resluts_path + 'GPP comparison of all models' + '_scenario 16' + '.png', dpi=300)
    # Display the plot
    plt.show()