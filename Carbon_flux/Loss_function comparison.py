
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import matplotlib.patches as patches

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

def bar_multi(metrics_data, variables, models, Y_metrics):
    plt.rcParams['font.family'] = 'Arial'
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    bar_width = 0.3
    outputs = ['Output1', 'Output2', 'Output3']

    # Modify the label for the 'Physics-aware transformer '
    modified_models = ['Physics-aware\ntransformer' if model == 'Physics-aware transformer' else model for model in
                       models]

    for i, metric_data in enumerate(metrics_data):
        ax = axes[i]
        for k, variable in enumerate(variables):
            for j, model in enumerate(modified_models):
                x = k * len(models) + j
                values = metric_data[(variable, models[j])]
                for m, output in enumerate(outputs):
                    label_offset = 0.001 if values[m] >= 0 else -0.015
                    # Define lighter colors, as mentioned in your function
                    # lighter_greens = ['#262626', '#293890', '#BF1D2D']
                    # lighter_greens = ['black', '#589FF3', '#F94141']
                    # colors1 = ['black', 'blue', 'red']  # Define colors for plots
                    # colors2 = ['black', '#589FF3', '#F94141']  # Lighter colors
                    # lighter_greens = []
                    # for c1, c2 in zip(colors1, colors2):
                    #     rgb1 = mcolors.to_rgb(c1)
                    #     rgb2 = mcolors.to_rgb(c2)
                    #     new_rgb = [(r1 + r2) / 2 for r1, r2 in zip(rgb1, rgb2)]
                    #     lighter_greens.append(mcolors.to_hex(new_rgb))
                    # lighter_greens = ['#262626', '#23B2E0', '#EF232B']
                    # lighter_greens = ['#7A7A7A', '#5EB0D2', '#D93A3B']
                    # 中#7A7A7A 是一种中灰色。#5EB0D2 是一个较深的蓝绿色。 #D93A3B 是一个适中的红色
                    lighter_greens = ['#A0A0A0', '#8EC4D4', '#F36B6F']
                    # 浅
                    # lighter_greens = ['black', 'blue','red']
                    ax.bar(x + m * bar_width, values[m], bar_width, label=output, color=lighter_greens[m])
                    # Format the bar value to two decimal places
                    ax.text(x + m * bar_width, values[m] + label_offset, f'{values[m]:.2f}', ha='center', va='bottom',fontsize=12)
                if k < len(variables) - 1:
                    ax.axvline((k + 1) * len(models) - 0.2, color='black', linestyle='--', linewidth=1)
            # Add variable labels
            if i == 0:
                ax.text(k * len(models) + 1, 1.75, variable, fontsize=16, ha='center', va='center') # with meteorological
                # ax.text(k * len(models) + 1, 1.78, variable, fontsize=16, ha='center',va='center')  # without meteorological
        ax.tick_params(axis='y', labelsize=13)  # Sets the font size for both x and y axis tick labels
        ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
        ax.set_xticklabels([model for model in modified_models] * len(variables), fontsize=15)
        ax.set_ylabel(Y_metrics[i], fontsize=16, fontname='Arial')

        # Set y-limits based on the row
        if i == 0:  # 第一行
            ax.set_ylim(1.00, max(max(values) for values in metric_data.values())+0.05)  # Correct this line
        elif i == 1:  # 第二行
            ax.set_ylim(0.35, max(max(values) for values in metric_data.values())+0.05)  # with meteorological data
            # ax.set_ylim(0.25, max(max(values) for values in metric_data.values()) + 0.05)  # without meteorological data
        elif i == 2:  # 第三行
        # ax.set_ylim(0, max(max(values) for values in metric_data.values()) + 0.05)  # 示例设置
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置小数点格式为两位

        # handles, labels = ax.get_legend_handles_labels()
        # unique_labels = list(dict.fromkeys(labels))  # Remove duplicate labels
        # unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
        # ax.legend(handles=unique_handles, labels=unique_labels, loc='lower right')

    plt.tight_layout()
    plt.show()

    results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(results_path + 'Three output_RLM_0302_v1' + '.png', dpi=300)


def calculate_metrics(x, y):
    """
    计算 x 和 y 之间的 RMSE（均方根误差）、CC（皮尔逊相关系数）和 MAE（绝对值偏差）

    参数:
    x, y -- 两个等长的 numpy 数组

    返回:
    rmse -- 均方根误差
    cc -- 皮尔逊相关系数
    mae -- 平均绝对误差
    """
    x = np.array(x).flatten()  # 确保是一维数组
    y = np.array(y).flatten()

    # 检查数组长度，避免 np.corrcoef 出错
    if len(x) <= 1 or len(y) <= 1:
        raise ValueError("输入数据长度必须大于 1")
    # 计算 RMSE
    rmse = np.sqrt(np.mean((x - y) ** 2))
    # 计算 Pearson 相关系数（CC）
    cc = np.corrcoef(x, y)[0, 1]
    # 计算（偏差绝对值）
    bias = np.abs(np.mean(x - y))

    return rmse, cc, bias

# 给定影像的根路径
if __name__ == "__main__":
    # download the data
    inputdata_path = './data/sites_results/'
    # inputdata_path = './data/sites_sensitivity/'
    # linear dl
    #***************************************
    # DL physics-aware 深度学习三个输出
    DL_Y_predict_RL_0 = np.load(inputdata_path + 'DL_RL_predict_3.npy', allow_pickle=True)
    DL_Y_true_RL_0 = np.load(inputdata_path + 'DL_RL_true_3.npy', allow_pickle=True)
    DL_All_true_RL_0 = np.load(inputdata_path + 'DL_RL_sites_3.npy', allow_pickle=True)
    inter_mask_RL_0 = np.load(inputdata_path + 'DL_RL_x_mask_3.npy', allow_pickle=True)
    DL_y_qa_RL_0 = np.load(inputdata_path + 'DL_RL_y_qa_3.npy', allow_pickle=True)

    DL_Y_predict_RL_huber = np.load(inputdata_path + 'DL_RL_predict_huber.npy', allow_pickle=True)
    DL_Y_true_RL_huber = np.load(inputdata_path + 'DL_RL_true_huber.npy', allow_pickle=True)
    DL_All_true_RL_huber = np.load(inputdata_path + 'DL_RL_sites_huber.npy', allow_pickle=True)
    inter_mask_RL_huber = np.load(inputdata_path + 'DL_RL_x_mask_huber.npy', allow_pickle=True)
    DL_y_qa_RL_huber = np.load(inputdata_path + 'DL_RL_y_qa_huber.npy', allow_pickle=True)

    DL_Y_predict_RL_MAE = np.load(inputdata_path + 'DL_RL_predict_MAE.npy', allow_pickle=True)
    DL_Y_true_RL_MAE = np.load(inputdata_path + 'DL_RL_true_MAE.npy', allow_pickle=True)
    DL_All_true_RL_MAE = np.load(inputdata_path + 'DL_RL_sites_MAE.npy', allow_pickle=True)
    inter_mask_RL_MAE = np.load(inputdata_path + 'DL_RL_x_mask_MAE.npy', allow_pickle=True)
    DL_y_qa_RL_MAE = np.load(inputdata_path + 'DL_RL_y_qa_MAE.npy', allow_pickle=True)

    # linear dl
    DL_Y_predict_RLM_0 = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    DL_Y_true_RLM_0 = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    DL_All_true_RLM_0 = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    inter_mask_RLM_0 = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    DL_y_qa_RLM_0 = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)

    DL_Y_predict_RLM_huber = np.load(inputdata_path + 'DL_RLM_predict_huber.npy', allow_pickle=True)
    DL_Y_true_RLM_huber = np.load(inputdata_path + 'DL_RLM_true_huber.npy', allow_pickle=True)
    DL_All_true_RLM_huber = np.load(inputdata_path + 'DL_RLM_sites_huber.npy', allow_pickle=True)
    inter_mask_RLM_huber = np.load(inputdata_path + 'DL_RLM_x_mask_huber.npy', allow_pickle=True)
    DL_y_qa_RLM_huber = np.load(inputdata_path + 'DL_RLM_y_qa_huber.npy', allow_pickle=True)

    DL_Y_predict_RLM_MAE = np.load(inputdata_path + 'DL_RLM_predict_MAE_1.npy', allow_pickle=True)
    DL_Y_true_RLM_MAE = np.load(inputdata_path + 'DL_RLM_true_MAE_1.npy', allow_pickle=True)
    DL_All_true_RLM_MAE = np.load(inputdata_path + 'DL_RLM_sites_MAE_1.npy', allow_pickle=True)
    inter_mask_RLM_MAE = np.load(inputdata_path + 'DL_RLM_x_mask_MAE_1.npy', allow_pickle=True)
    DL_y_qa_RLM_MAE = np.load(inputdata_path + 'DL_RLM_y_qa_MAE_1.npy', allow_pickle=True)

    DL_NEE_0_RLM = DL_Y_predict_RLM_0[:, :, 0]
    DL_NEE_huber_RLM = DL_Y_predict_RLM_huber[:, :, 0]
    DL_NEE_MAE_RLM = DL_Y_predict_RLM_MAE[:, :, 0]

    Ground_NEE_0_RLM = DL_Y_true_RLM_0[:, :, 0]
    Ground_NEE_huber_RLM = DL_Y_true_RLM_huber[:, :, 0]
    Ground_NEE_MAE_RLM = DL_Y_true_RLM_MAE[:, :, 0]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_NEE_RLM_0 = Ground_NEE_0_RLM != -9999
    mask_NEE_RLM_huber = Ground_NEE_huber_RLM != -9999
    mask_NEE_RLM_MAE = Ground_NEE_MAE_RLM != -9999

    DL_NEE_0_RLM_filtered = DL_NEE_0_RLM[mask_NEE_RLM_0]
    DL_NEE_huber_RLM_filtered = DL_NEE_huber_RLM[mask_NEE_RLM_huber]
    DL_NEE_MAE_RLM_filtered = DL_NEE_MAE_RLM[mask_NEE_RLM_MAE]

    Ground_NEE_0_RLM_filtered = Ground_NEE_0_RLM[mask_NEE_RLM_0]
    Ground_NEE_huber_RLM_filtered = Ground_NEE_huber_RLM[mask_NEE_RLM_huber]
    Ground_NEE_MAE_RLM_filtered = Ground_NEE_MAE_RLM[mask_NEE_RLM_MAE]

    # 没有气象数据的
    DL_NEE_0_RL = DL_Y_predict_RL_0[:, :, 0]
    DL_NEE_huber_RL = DL_Y_predict_RL_huber[:, :, 0]
    DL_NEE_MAE_RL = DL_Y_predict_RL_MAE[:, :, 0]

    Ground_NEE_0_RL = DL_Y_true_RL_0[:, :, 0]
    Ground_NEE_huber_RL = DL_Y_true_RL_huber[:, :, 0]
    Ground_NEE_MAE_RL = DL_Y_true_RL_MAE[:, :, 0]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step huber: 创建掩码（值不等于 -9999）**
    mask_NEE_RL_0 = Ground_NEE_0_RL != -9999
    mask_NEE_RL_huber = Ground_NEE_huber_RL != -9999
    mask_NEE_RL_MAE = Ground_NEE_MAE_RL != -9999

    DL_NEE_0_RL_filtered = DL_NEE_0_RL[mask_NEE_RL_0]
    DL_NEE_huber_RL_filtered = DL_NEE_huber_RL[mask_NEE_RL_huber]
    DL_NEE_MAE_RL_filtered = DL_NEE_MAE_RL[mask_NEE_RL_MAE]

    Ground_NEE_0_RL_filtered = Ground_NEE_0_RL[mask_NEE_RL_0]
    Ground_NEE_huber_RL_filtered = Ground_NEE_huber_RL[mask_NEE_RL_huber]
    Ground_NEE_MAE_RL_filtered = Ground_NEE_MAE_RL[mask_NEE_RL_MAE]

    DL_GPP_0_RLM = DL_Y_predict_RLM_0[:, :, 1]
    DL_GPP_huber_RLM = DL_Y_predict_RLM_huber[:, :, 1]
    DL_GPP_MAE_RLM = DL_Y_predict_RLM_MAE[:, :, 1]

    Ground_GPP_0_RLM = DL_Y_true_RLM_0[:, :, 1]
    Ground_GPP_huber_RLM = DL_Y_true_RLM_huber[:, :, 1]
    Ground_GPP_MAE_RLM = DL_Y_true_RLM_MAE[:, :, 1]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_GPP_0 = Ground_GPP_0_RLM != -9999
    mask_GPP_huber = Ground_GPP_huber_RLM != -9999
    mask_GPP_MAE = Ground_GPP_MAE_RLM != -9999

    DL_GPP_0_RLM_filtered = DL_GPP_0_RLM[mask_GPP_0]
    DL_GPP_huber_RLM_filtered = DL_GPP_huber_RLM[mask_GPP_huber]
    DL_GPP_MAE_RLM_filtered = DL_GPP_MAE_RLM[mask_GPP_MAE]

    Ground_GPP_0_RLM_filtered = Ground_GPP_0_RLM[mask_GPP_0]
    Ground_GPP_huber_RLM_filtered = Ground_GPP_huber_RLM[mask_GPP_huber]
    Ground_GPP_MAE_RLM_filtered = Ground_GPP_MAE_RLM[mask_GPP_MAE]

    # 没有气象数据的
    DL_GPP_0_RL = DL_Y_predict_RL_0[:, :, 1]
    DL_GPP_huber_RL = DL_Y_predict_RL_huber[:, :, 1]
    DL_GPP_MAE_RL = DL_Y_predict_RL_MAE[:, :, 1]

    Ground_GPP_0_RL = DL_Y_true_RL_0[:, :, 1]
    Ground_GPP_huber_RL = DL_Y_true_RL_huber[:, :, 1]
    Ground_GPP_MAE_RL = DL_Y_true_RL_MAE[:, :, 1]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_RL_GPP_0 = Ground_GPP_0_RL != -9999
    mask_RL_GPP_huber = Ground_GPP_huber_RL != -9999
    mask_RL_GPP_MAE = Ground_GPP_MAE_RL != -9999

    DL_GPP_0_RL_filtered = DL_GPP_0_RL[mask_RL_GPP_0]
    DL_GPP_huber_RL_filtered = DL_GPP_huber_RL[mask_RL_GPP_huber]
    DL_GPP_MAE_RL_filtered = DL_GPP_MAE_RL[mask_RL_GPP_MAE]

    Ground_GPP_0_RL_filtered = Ground_GPP_0_RL[mask_RL_GPP_0]
    Ground_GPP_huber_RL_filtered = Ground_GPP_huber_RL[mask_RL_GPP_huber]
    Ground_GPP_MAE_RL_filtered = Ground_GPP_MAE_RL[mask_RL_GPP_MAE]

    DL_RECO_0_RLM = DL_Y_predict_RLM_0[:, :, 2]
    DL_RECO_huber_RLM = DL_Y_predict_RLM_huber[:, :, 2]
    DL_RECO_MAE_RLM = DL_Y_predict_RLM_MAE[:, :, 2]

    Ground_RECO_0_RLM = DL_Y_true_RLM_0[:, :, 2]
    Ground_RECO_huber_RLM = DL_Y_true_RLM_huber[:, :, 2]
    Ground_RECO_MAE_RLM = DL_Y_true_RLM_MAE[:, :, 2]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step huber: 创建掩码（值不等于 -9999）**
    mask_RECO_0 = Ground_RECO_0_RLM != -9999
    mask_RECO_huber = Ground_RECO_huber_RLM != -9999
    mask_RECO_MAE = Ground_RECO_MAE_RLM != -9999

    DL_RECO_0_RLM_filtered = DL_RECO_0_RLM[mask_RECO_0]
    DL_RECO_huber_RLM_filtered = DL_RECO_huber_RLM[mask_RECO_huber]
    DL_RECO_MAE_RLM_filtered = DL_RECO_MAE_RLM[mask_RECO_MAE]

    Ground_RECO_0_RLM_filtered = Ground_RECO_0_RLM[mask_RECO_0]
    Ground_RECO_huber_RLM_filtered = Ground_RECO_huber_RLM[mask_RECO_huber]
    Ground_RECO_MAE_RLM_filtered = Ground_RECO_MAE_RLM[mask_RECO_MAE]

    # 没有气象数据的
    DL_RECO_0_RL = DL_Y_predict_RL_0[:, :, 2]
    DL_RECO_huber_RL = DL_Y_predict_RL_huber[:, :, 2]
    DL_RECO_MAE_RL = DL_Y_predict_RL_MAE[:, :, 2]

    Ground_RECO_0_RL = DL_Y_true_RL_0[:, :, 2]
    Ground_RECO_huber_RL = DL_Y_true_RL_huber[:, :, 2]
    Ground_RECO_MAE_RL = DL_Y_true_RL_MAE[:, :, 2]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step huber: 创建掩码（值不等于 -9999）**
    mask_RL_RECO_0 = Ground_RECO_0_RL != -9999
    mask_RL_RECO_huber = Ground_RECO_huber_RL != -9999
    mask_RL_RECO_MAE = Ground_RECO_MAE_RL != -9999

    DL_RECO_0_RL_filtered = DL_RECO_0_RL[mask_RL_RECO_0]
    DL_RECO_huber_RL_filtered = DL_RECO_huber_RL[mask_RL_RECO_huber]
    DL_RECO_MAE_RL_filtered = DL_RECO_MAE_RL[mask_RL_RECO_MAE]

    Ground_RECO_0_RL_filtered = Ground_RECO_0_RL[mask_RL_RECO_0]
    Ground_RECO_huber_RL_filtered = Ground_RECO_huber_RL[mask_RL_RECO_huber]
    Ground_RECO_MAE_RL_filtered = Ground_RECO_MAE_RL[mask_RL_RECO_MAE]

    # 使用气象数据计算的值 深度学习
    DL_NEE_0_RLM_rmse, DL_NEE_0_RLM_cc, DL_NEE_0_RLM_bias = calculate_metrics(Ground_NEE_0_RLM_filtered,
                                                                              DL_NEE_0_RLM_filtered)
    DL_NEE_1_RLM_rmse, DL_NEE_1_RLM_cc, DL_NEE_1_RLM_bias = calculate_metrics(Ground_NEE_huber_RLM_filtered,
                                                                              DL_NEE_huber_RLM_filtered)
    DL_NEE_MAE_RLM_rmse, DL_NEE_MAE_RLM_cc, DL_NEE_MAE_RLM_bias = calculate_metrics(Ground_NEE_MAE_RLM_filtered,
                                                                              DL_NEE_MAE_RLM_filtered)

    DL_GPP_0_RLM_rmse, DL_GPP_0_RLM_cc, DL_GPP_0_RLM_bias = calculate_metrics(Ground_GPP_0_RLM_filtered,
                                                                              DL_GPP_0_RLM_filtered)
    DL_GPP_huber_RLM_rmse, DL_GPP_huber_RLM_cc, DL_GPP_huber_RLM_bias = calculate_metrics(Ground_GPP_0_RLM_filtered,
                                                                              DL_GPP_0_RLM_filtered)
    DL_GPP_MAE_RLM_rmse, DL_GPP_MAE_RLM_cc, DL_GPP_MAE_RLM_bias = calculate_metrics(Ground_GPP_MAE_RLM_filtered,
                                                                              DL_GPP_MAE_RLM_filtered)
    DL_RECO_0_RLM_rmse, DL_RECO_0_RLM_cc, DL_RECO_0_RLM_bias = calculate_metrics(Ground_RECO_0_RLM_filtered,
                                                                                 DL_RECO_0_RLM_filtered)
    DL_RECO_huber_RLM_rmse, DL_RECO_huber_RLM_cc, DL_RECO_huber_RLM_bias = calculate_metrics(Ground_RECO_huber_RLM_filtered,
                                                                                 DL_RECO_huber_RLM_filtered)
    DL_RECO_MAE_RLM_rmse, DL_RECO_MAE_RLM_cc, DL_RECO_MAE_RLM_bias = calculate_metrics(Ground_RECO_MAE_RLM_filtered,
                                                                                 DL_RECO_MAE_RLM_filtered)

    X_variables = ['Ground_RECO_0_RLM_filtered', 'Ground_RECO_huber_RLM_filtered', 'Ground_RECO_MAE_RLM_filtered',
                   'Ground_RECO_0_RL_filtered','Ground_RECO_huber_RL_filtered', 'Ground_RECO_MAE_RL_filtered']
    Y_variables = ['DL_RECO_0_RLM_filtered', 'DL_RECO_huber_RLM_filtered', 'DL_RECO_MAE_RLM_filtered',
                   'DL_RECO_0_RL_filtered', 'DL_RECO_huber_RL_filtered', 'DL_RECO_MAE_RL_filtered']

    model_subtitle = ['Physics-aware transformer\nwith RMSE loss', 'Physics-aware transformer\nwith Huber loss',
                      'Physics-aware transformer with\nwith MAE loss']  # 'CFR without physical constraints',
    # model_subtitle = ['RECO', 'RECO', 'RECO']  # 'CFR without physical constraints',
    ax_subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # ,'(g)','(h)'
    import matplotlib.pyplot as plt

    # 设置共享colorbar位置
    # divider = make_axes_locatable(axes[1, 3])
    # cax = divider.append_axes("right", size="5%", pad=0.1)

    # 设置字体
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.family'] = 'Arial'
    # 创建图形和子图
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    # 找到x和y的总体最小值和最大值
    overall_x_min = min(globals()[var].min() for var in X_variables)
    overall_x_max = max(globals()[var].max() for var in X_variables)
    overall_y_min = min(globals()[var].min() for var in Y_variables)
    overall_y_max = max(globals()[var].max() for var in Y_variables)

    # 初始化公共坐标轴范围
    common_min = -3
    common_max = 3
    # common_min = 0
    # common_max = 8
    # 颜色定义
    colors = [
        'white', 'lightcyan', 'lightblue', 'deepskyblue', 'blue',
        'mediumseagreen', 'green', 'greenyellow', 'yellow', 'gold',
        'orange', 'tomato', 'red', 'firebrick', 'darkred'
    ]

    n_bins = len(colors)
    first_bounds = None  # 用于存储第一个子图的边界

    for i in range(6):
        row = i // 3
        col = i % 3
        x_variable = X_variables[i]
        y_variable = Y_variables[i]
        x = globals()[x_variable]
        y = globals()[y_variable]

        # 计算统计数据
        rmse = np.sqrt(mean_squared_error(x, y))
        cc = np.corrcoef(x, y)[0, 1]
        n = len(x)

        # 创建直方图密度
        bins_n = 100
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins_n)
        density_values = hist.ravel()

        # 只计算第一个子图的边界
        if first_bounds is None:
            max_density = hist.max()
            first_bounds = np.logspace(np.log10(1), np.log10(max_density + 1), n_bins + 1)
            print("First Bounds:", [f"{b:.2f}" for b in first_bounds])

        bounds = first_bounds  # 对所有子图使用相同的边界

        # 根据边界过滤颜色
        filtered_colors = ['white' if bound < bounds[5] else color for color, bound in zip(colors, bounds[:n_bins])]
        cmap = ListedColormap(filtered_colors)
        norm = BoundaryNorm(bounds, cmap.N)

        # 绘制直方图
        h = axs[row, col].hist2d(x, y, bins=bins_n, cmap=cmap, norm=norm)

        # 在图中显示RMSE和CC
        # text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}\nn = {int(n)}"
        text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}"
        axs[row, col].text(0.05, 0.97, text, verticalalignment='top', transform=axs[row, col].transAxes,
                           color='red', fontweight='bold', fontsize=16)

        # 绘制对角线
        axs[row, col].plot([common_min, common_max], [common_min, common_max], color='#535CA8', linestyle='--',
                           linewidth=1)
        if col == 0:
            common_min = 0
            common_max = 8
        elif col == 1:
            common_min = 0
            common_max = 8
        else:
            common_min = 0
            common_max = 8

        # 设置坐标轴范围、刻度和标签
        axs[row, col].set_xlim([common_min, common_max])
        axs[row, col].set_ylim([common_min, common_max])

        num_ticks = 5
        ticks_x = np.linspace(common_min, common_max, num_ticks)
        ticks_y = np.linspace(common_min, common_max, num_ticks)
        axs[row, col].set_xticks(np.round(ticks_x).astype(int))
        axs[row, col].set_yticks(np.round(ticks_y).astype(int))
        axs[row, col].tick_params(axis='both', labelsize=16)

        axs[row, col].set_xlabel('Flux tower', fontsize=18)
        axs[row, col].set_ylabel('Model estimated', fontsize=18)

        if row == 0:
            axs[row, col].set_title(model_subtitle[i], fontsize=18)
        # 添加 ax_subtitle 到右下方
        axs[row, col].text(0.95, 0.05, ax_subtitle[i], ha='right', va='bottom', transform=axs[row, col].transAxes,
                           fontsize=18)

    # 创建图例位置
    legend_position = [0.92, 0.15, 0.015, 0.75]  # 自定义位置

    # 创建一个新的图形用于渐变图例
    legend_ax = fig.add_axes(legend_position)  # 创建图例轴

    # 渐变图例条带
    n_color_bands = len(colors[5:])  # 计算要展示的颜色条带数
    for i, color in enumerate(filtered_colors[5:]):
        # 创建渐变的 Patch（条带）
        rectangle = mpatches.Rectangle((0, i), 1, 1, color=color)  # 每个条带的大小
        legend_ax.add_patch(rectangle)

    # 设置图例y轴的刻度位置和标签
    y_ticks = np.arange(n_color_bands + 1)  # 增加一个刻度位置，使标签比颜色条多一个
    y_labels = [f"{bound:.0f}" for bound in bounds[5:]]  # 添加一个标签以包含最后的边界

    legend_ax.set_yticks(y_ticks)  # 设置y轴刻度位置
    legend_ax.set_yticklabels(y_labels, ha='left', fontsize=16)  # 设置y轴刻度标签并左对齐

    # 将刻度设置在右侧
    legend_ax.yaxis.set_ticks_position('right')

    # 隐藏图例轴的x轴和边框
    legend_ax.set_xticks([])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, n_color_bands)

    # 添加图例标题
    legend_ax.set_title("Counts", fontsize=18)

    # 显示图形
    # 设置整体标签
    fig.text(0.5, 0.05, 'RECO (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center', fontsize=18,
             fontname='Arial')
    fig.text(0.03, 0.7, 'Input with\nmeteorological data', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Arial')
    fig.text(0.03, 0.3, 'Input without\nmeteorological data', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Arial')
    # fig.text(0.03, 0.7, 'Input with\nsoil content', ha='center', va='center', rotation='vertical',
    #          fontsize=18, fontname='Arial')
    # fig.text(0.03, 0.3, 'Input without\nsoil content', ha='center', va='center', rotation='vertical',
    #          fontsize=18, fontname='Arial')
    # 调整整体布局
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.35, hspace=0.35)

    # base_name = 'Comparison of predicted RECO by the CFR model and machine learning models using RLFM(1) or RLF(2) as input'
    # plt.suptitle(base_name, ha='center', fontsize=12, y=0.99, fontname='Arial')

    resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(resluts_path + 'RECO three loss' + '.png', dpi=300)
    # Display the plot
    plt.show()

