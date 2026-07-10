import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
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
                ax.text(k * len(models) + 1, 1.85, variable, fontsize=16, ha='center', va='center') # with meteorological
                # ax.text(k * len(models) + 1, 1.78, variable, fontsize=16, ha='center',va='center')  # without meteorological
        ax.tick_params(axis='y', labelsize=13)  # Sets the font size for both x and y axis tick labels
        ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
        ax.set_xticklabels([model for model in modified_models] * len(variables), fontsize=15)
        ax.set_ylabel(Y_metrics[i], fontsize=16, fontname='Arial')

        # Set y-limits based on the row
        if i == 0:  # 第一行
            # ax.set_ylim(1.00, max(max(values) for values in metric_data.values())+0.05)  # Correct this line
            ax.set_ylim(1.00, 1.8)  # Correct this line
        elif i == 1:  # 第二行
            # ax.set_ylim(0.3, max(max(values) for values in metric_data.values())+0.05)  # with meteorological data
            ax.set_ylim(0.25, 0.85)  # with meteorological data
            # ax.set_ylim(0.25, max(max(values) for values in metric_data.values()) + 0.05)  # without meteorological data
        elif i == 2:  # 第三行
        # ax.set_ylim(0, max(max(values) for values in metric_data.values()) + 0.05)  # 示例设置
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置小数点格式为两位
              ax.set_ylim(0.00, 0.25)

        # handles, labels = ax.get_legend_handles_labels()
        # unique_labels = list(dict.fromkeys(labels))  # Remove duplicate labels
        # unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
        # ax.legend(handles=unique_handles, labels=unique_labels, loc='lower right')

    plt.tight_layout()


    results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(results_path + 'Three output_RLM 0404' + '.png', dpi=300)
    plt.show()

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
    # inputdata_path = './data/sites_results/'
    inputdata_path = './data/sites_sensitivity/'
    # linear dl
    #***************************************
    # DL physics-aware 深度学习三个输出
    DL_Y_predict_RL_0 = np.load(inputdata_path + 'DL_RL_predict_3.npy', allow_pickle=True)
    DL_Y_true_RL_0 = np.load(inputdata_path + 'DL_RL_true_3.npy', allow_pickle=True)
    DL_All_true_RL_0 = np.load(inputdata_path + 'DL_RL_sites_3.npy', allow_pickle=True)
    inter_mask_RL_0 = np.load(inputdata_path + 'DL_RL_x_mask_3.npy', allow_pickle=True)
    DL_y_qa_RL_0 = np.load(inputdata_path + 'DL_RL_y_qa_3.npy', allow_pickle=True)

    DL_Y_predict_RL_1 = np.load(inputdata_path + 'DL_RL_predict_good_modis.npy', allow_pickle=True)
    DL_Y_true_RL_1 = np.load(inputdata_path + 'DL_RL_true_good_modis.npy', allow_pickle=True)
    DL_All_true_RL_1 = np.load(inputdata_path + 'DL_RL_sites_good_modis.npy', allow_pickle=True)
    inter_mask_RL_1 = np.load(inputdata_path + 'DL_RL_x_mask_good_modis.npy', allow_pickle=True)
    DL_y_qa_RL_1 = np.load(inputdata_path + 'DL_RL_y_qa_good_modis.npy', allow_pickle=True)

    DL_Y_predict_RL_2 = np.load(inputdata_path + 'DL_RL_predict_good_y.npy', allow_pickle=True)
    DL_Y_true_RL_2 = np.load(inputdata_path + 'DL_RL_true_good_y.npy', allow_pickle=True)
    DL_All_true_RL_2 = np.load(inputdata_path + 'DL_RL_sites_good_y.npy', allow_pickle=True)
    inter_mask_RL_2 = np.load(inputdata_path + 'DL_RL_x_mask_good_y.npy', allow_pickle=True)
    DL_y_qa_RL_2 = np.load(inputdata_path + 'DL_RL_y_qa_good_y.npy', allow_pickle=True)

    # linear dl
    DL_Y_predict_RLM_0 = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    DL_Y_true_RLM_0 = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    DL_All_true_RLM_0 = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    inter_mask_RLM_0 = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    DL_y_qa_RLM_0 = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)

    DL_Y_predict_RLM_1 = np.load(inputdata_path + 'DL_RLM_predict_good_modis.npy', allow_pickle=True)
    DL_Y_true_RLM_1 = np.load(inputdata_path + 'DL_RLM_true_good_modis.npy', allow_pickle=True)
    DL_All_true_RLM_1 = np.load(inputdata_path + 'DL_RLM_sites_good_modis.npy', allow_pickle=True)
    inter_mask_RLM_1 = np.load(inputdata_path + 'DL_RLM_x_mask_good_modis.npy', allow_pickle=True)
    DL_y_qa_RLM_1 = np.load(inputdata_path + 'DL_RLM_y_qa_good_modis.npy', allow_pickle=True)

    DL_Y_predict_RLM_2 = np.load(inputdata_path + 'DL_RLM_predict_good_y.npy', allow_pickle=True)
    DL_Y_true_RLM_2 = np.load(inputdata_path + 'DL_RLM_true_good_y.npy', allow_pickle=True)
    DL_All_true_RLM_2 = np.load(inputdata_path + 'DL_RLM_sites_good_y.npy', allow_pickle=True)
    inter_mask_RLM_2 = np.load(inputdata_path + 'DL_RLM_x_mask_good_y.npy', allow_pickle=True)
    DL_y_qa_RLM_2 = np.load(inputdata_path + 'DL_RLM_y_qa_good_y.npy', allow_pickle=True)
    inter_mask_RLM_2[:, :, 0, 1, 1]

    # print(min(XGB_GPP_predict),min(XGB_GPP_predict),min(XGB_GPP_predict_RLM),min(XGB_GPP_predict_RLM))
    # print(min(XGB_RECO_predict), min(XGB_RECO_predict), min(XGB_RECO_predict_RLM), min(XGB_RECO_predict_RLM))
    DL_NEE_0_RLM = DL_Y_predict_RLM_0[:, :, 0]
    DL_NEE_1_RLM = DL_Y_predict_RLM_1[:, :, 0]
    DL_NEE_2_RLM = DL_Y_predict_RLM_2[:, :, 0]

    Ground_NEE_0_RLM = DL_Y_true_RLM_0[:, :, 0]
    Ground_NEE_1_RLM = DL_Y_true_RLM_1[:, :, 0]
    Ground_NEE_2_RLM = DL_Y_true_RLM_2[:, :, 0]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_NEE_RLM_0 = Ground_NEE_0_RLM != -9999
    mask_NEE_RLM_1 = Ground_NEE_1_RLM != -9999
    mask_NEE_RLM_2 = Ground_NEE_2_RLM != -9999

    DL_NEE_0_RLM_filtered = DL_NEE_0_RLM[mask_NEE_RLM_0]
    DL_NEE_1_RLM_filtered = DL_NEE_1_RLM[mask_NEE_RLM_1]
    DL_NEE_2_RLM_filtered = DL_NEE_2_RLM[mask_NEE_RLM_2]

    Ground_NEE_0_RLM_filtered = Ground_NEE_0_RLM[mask_NEE_RLM_0]
    Ground_NEE_1_RLM_filtered = Ground_NEE_1_RLM[mask_NEE_RLM_1]
    Ground_NEE_2_RLM_filtered = Ground_NEE_2_RLM[mask_NEE_RLM_2]

    # 没有气象数据的
    DL_NEE_0_RL = DL_Y_predict_RL_0[:, :, 0]
    DL_NEE_1_RL = DL_Y_predict_RL_1[:, :, 0]
    DL_NEE_2_RL = DL_Y_predict_RL_2[:, :, 0]

    Ground_NEE_0_RL = DL_Y_true_RL_0[:, :, 0]
    Ground_NEE_1_RL = DL_Y_true_RL_1[:, :, 0]
    Ground_NEE_2_RL = DL_Y_true_RL_2[:, :, 0]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_NEE_RL_0 = Ground_NEE_0_RL != -9999
    mask_NEE_RL_1 = Ground_NEE_1_RL != -9999
    mask_NEE_RL_2 = Ground_NEE_2_RL != -9999

    DL_NEE_0_RL_filtered = DL_NEE_0_RL[mask_NEE_RL_0]
    DL_NEE_1_RL_filtered = DL_NEE_1_RL[mask_NEE_RL_1]
    DL_NEE_2_RL_filtered = DL_NEE_2_RL[mask_NEE_RL_2]

    Ground_NEE_0_RL_filtered = Ground_NEE_0_RL[mask_NEE_RL_0]
    Ground_NEE_1_RL_filtered = Ground_NEE_1_RL[mask_NEE_RL_1]
    Ground_NEE_2_RL_filtered = Ground_NEE_2_RL[mask_NEE_RL_2]

    DL_GPP_0_RLM = DL_Y_predict_RLM_0[:, :, 1]
    DL_GPP_1_RLM = DL_Y_predict_RLM_1[:, :, 1]
    DL_GPP_2_RLM = DL_Y_predict_RLM_2[:, :, 1]

    Ground_GPP_0_RLM = DL_Y_true_RLM_0[:, :, 1]
    Ground_GPP_1_RLM = DL_Y_true_RLM_1[:, :, 1]
    Ground_GPP_2_RLM = DL_Y_true_RLM_2[:, :, 1]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_GPP_0 = Ground_GPP_0_RLM != -9999
    mask_GPP_1 = Ground_GPP_1_RLM != -9999
    mask_GPP_2 = Ground_GPP_2_RLM != -9999

    DL_GPP_0_RLM_filtered = DL_GPP_0_RLM[mask_GPP_0]
    DL_GPP_1_RLM_filtered = DL_GPP_1_RLM[mask_GPP_1]
    DL_GPP_2_RLM_filtered = DL_GPP_2_RLM[mask_GPP_2]

    Ground_GPP_0_RLM_filtered = Ground_GPP_0_RLM[mask_GPP_0]
    Ground_GPP_1_RLM_filtered = Ground_GPP_1_RLM[mask_GPP_1]
    Ground_GPP_2_RLM_filtered = Ground_GPP_2_RLM[mask_GPP_2]

    # 没有气象数据的
    DL_GPP_0_RL = DL_Y_predict_RL_0[:, :, 1]
    DL_GPP_1_RL = DL_Y_predict_RL_1[:, :, 1]
    DL_GPP_2_RL = DL_Y_predict_RL_2[:, :, 1]

    Ground_GPP_0_RL = DL_Y_true_RL_0[:, :, 1]
    Ground_GPP_1_RL = DL_Y_true_RL_1[:, :, 1]
    Ground_GPP_2_RL = DL_Y_true_RL_2[:, :, 1]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_RL_GPP_0 = Ground_GPP_0_RL != -9999
    mask_RL_GPP_1 = Ground_GPP_1_RL != -9999
    mask_RL_GPP_2 = Ground_GPP_2_RL != -9999

    DL_GPP_0_RL_filtered = DL_GPP_0_RL[mask_RL_GPP_0]
    DL_GPP_1_RL_filtered = DL_GPP_1_RL[mask_RL_GPP_1]
    DL_GPP_2_RL_filtered = DL_GPP_2_RL[mask_RL_GPP_2]

    Ground_GPP_0_RL_filtered = Ground_GPP_0_RL[mask_RL_GPP_0]
    Ground_GPP_1_RL_filtered = Ground_GPP_1_RL[mask_RL_GPP_1]
    Ground_GPP_2_RL_filtered = Ground_GPP_2_RL[mask_RL_GPP_2]

    DL_RECO_0_RLM = DL_Y_predict_RLM_0[:, :, 2]
    DL_RECO_1_RLM = DL_Y_predict_RLM_1[:, :, 2]
    DL_RECO_2_RLM = DL_Y_predict_RLM_2[:, :, 2]

    Ground_RECO_0_RLM = DL_Y_true_RLM_0[:, :, 2]
    Ground_RECO_1_RLM = DL_Y_true_RLM_1[:, :, 2]
    Ground_RECO_2_RLM = DL_Y_true_RLM_2[:, :, 2]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_RECO_0 = Ground_RECO_0_RLM != -9999
    mask_RECO_1 = Ground_RECO_1_RLM != -9999
    mask_RECO_2 = Ground_RECO_2_RLM != -9999

    DL_RECO_0_RLM_filtered = DL_RECO_0_RLM[mask_RECO_0]
    DL_RECO_1_RLM_filtered = DL_RECO_1_RLM[mask_RECO_1]
    DL_RECO_2_RLM_filtered = DL_RECO_2_RLM[mask_RECO_2]

    Ground_RECO_0_RLM_filtered = Ground_RECO_0_RLM[mask_RECO_0]
    Ground_RECO_1_RLM_filtered = Ground_RECO_1_RLM[mask_RECO_1]
    Ground_RECO_2_RLM_filtered = Ground_RECO_2_RLM[mask_RECO_2]

    # 没有气象数据的
    DL_RECO_0_RL = DL_Y_predict_RL_0[:, :, 2]
    DL_RECO_1_RL = DL_Y_predict_RL_1[:, :, 2]
    DL_RECO_2_RL = DL_Y_predict_RL_2[:, :, 2]

    Ground_RECO_0_RL = DL_Y_true_RL_0[:, :, 2]
    Ground_RECO_1_RL = DL_Y_true_RL_1[:, :, 2]
    Ground_RECO_2_RL = DL_Y_true_RL_2[:, :, 2]
    # DL RLM单独预测的碳通量 没有气象变量
    # 第一组值:插值与不插值的一起
    # **Step 1: 创建掩码（值不等于 -9999）**
    mask_RL_RECO_0 = Ground_RECO_0_RL != -9999
    mask_RL_RECO_1 = Ground_RECO_1_RL != -9999
    mask_RL_RECO_2 = Ground_RECO_2_RL != -9999

    DL_RECO_0_RL_filtered = DL_RECO_0_RL[mask_RL_RECO_0]
    DL_RECO_1_RL_filtered = DL_RECO_1_RL[mask_RL_RECO_1]
    DL_RECO_2_RL_filtered = DL_RECO_2_RL[mask_RL_RECO_2]

    Ground_RECO_0_RL_filtered = Ground_RECO_0_RL[mask_RL_RECO_0]
    Ground_RECO_1_RL_filtered = Ground_RECO_1_RL[mask_RL_RECO_1]
    Ground_RECO_2_RL_filtered = Ground_RECO_2_RL[mask_RL_RECO_2]
    #***************************************
    # Physics-aware RF三个输出 有气象数据的
    # RF three  第一种情况 所有数据
    RF_NEE_predict_RLM_0 = np.load(inputdata_path + 'RF_RLM_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true_RLM_0 = np.load(inputdata_path + 'RF_RLM_NEE_true_3.npy', allow_pickle=True)

    RF_GPP_predict_RLM_0 = np.load(inputdata_path + 'RF_RLM_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true_RLM_0 = np.load(inputdata_path + 'RF_RLM_GPP_true_3.npy', allow_pickle=True)

    RF_RECO_predict_RLM_0 = np.load(inputdata_path + 'RF_RLM_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true_RLM_0 = np.load(inputdata_path + 'RF_RLM_RECO_true_3.npy', allow_pickle=True)

    RF_NEE_predict_RL_0 = np.load(inputdata_path + 'RF_RL_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true_RL_0 = np.load(inputdata_path + 'RF_RL_NEE_true_3.npy', allow_pickle=True)

    RF_GPP_predict_RL_0 = np.load(inputdata_path + 'RF_RL_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true_RL_0 = np.load(inputdata_path + 'RF_RL_GPP_true_3.npy', allow_pickle=True)

    RF_RECO_predict_RL_0 = np.load(inputdata_path + 'RF_RL_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true_RL_0 = np.load(inputdata_path + 'RF_RL_RECO_true_3.npy', allow_pickle=True)

    # RF three  第2种情况 只有质量好的modis数据
    RF_NEE_predict_RLM_1 = np.load(inputdata_path + 'RF_RLM_NEE_predict_good_modis.npy', allow_pickle=True)
    RF_NEE_true_RLM_1 = np.load(inputdata_path + 'RF_RLM_NEE_true_good_modis.npy', allow_pickle=True)

    RF_GPP_predict_RLM_1 = np.load(inputdata_path + 'RF_RLM_GPP_predict_good_modis.npy', allow_pickle=True)
    RF_GPP_true_RLM_1 = np.load(inputdata_path + 'RF_RLM_GPP_true_good_modis.npy', allow_pickle=True)

    RF_RECO_predict_RLM_1 = np.load(inputdata_path + 'RF_RLM_RECO_predict_good_modis.npy', allow_pickle=True)
    RF_RECO_true_RLM_1 = np.load(inputdata_path + 'RF_RLM_RECO_true_good_modis.npy', allow_pickle=True)

    RF_NEE_predict_RL_1 = np.load(inputdata_path + 'RF_RL_NEE_predict_good_modis.npy', allow_pickle=True)
    RF_NEE_true_RL_1 = np.load(inputdata_path + 'RF_RL_NEE_true_good_modis.npy', allow_pickle=True)

    RF_GPP_predict_RL_1 = np.load(inputdata_path + 'RF_RL_GPP_predict_good_modis.npy', allow_pickle=True)
    RF_GPP_true_RL_1 = np.load(inputdata_path + 'RF_RL_GPP_true_good_modis.npy', allow_pickle=True)

    RF_RECO_predict_RL_1 = np.load(inputdata_path + 'RF_RL_RECO_predict_good_modis.npy', allow_pickle=True)
    RF_RECO_true_RL_1 = np.load(inputdata_path + 'RF_RL_RECO_true_good_modis.npy', allow_pickle=True)

    # RF three  第2种情况 只有质量好的modis数据
    RF_NEE_predict_RLM_2 = np.load(inputdata_path + 'RF_RLM_NEE_predict_good_y.npy', allow_pickle=True)
    RF_NEE_true_RLM_2 = np.load(inputdata_path + 'RF_RLM_NEE_true_good_y.npy', allow_pickle=True)

    RF_GPP_predict_RLM_2 = np.load(inputdata_path + 'RF_RLM_GPP_predict_good_y.npy', allow_pickle=True)
    RF_GPP_true_RLM_2 = np.load(inputdata_path + 'RF_RLM_GPP_true_good_y.npy', allow_pickle=True)

    RF_RECO_predict_RLM_2 = np.load(inputdata_path + 'RF_RLM_RECO_predict_good_y.npy', allow_pickle=True)
    RF_RECO_true_RLM_2 = np.load(inputdata_path + 'RF_RLM_RECO_true_good_y.npy', allow_pickle=True)

    RF_NEE_predict_RL_2 = np.load(inputdata_path + 'RF_RL_NEE_predict_good_y.npy', allow_pickle=True)
    RF_NEE_true_RL_2 = np.load(inputdata_path + 'RF_RL_NEE_true_good_y.npy', allow_pickle=True)

    RF_GPP_predict_RL_2 = np.load(inputdata_path + 'RF_RL_GPP_predict_good_y.npy', allow_pickle=True)
    RF_GPP_true_RL_2 = np.load(inputdata_path + 'RF_RL_GPP_true_good_y.npy', allow_pickle=True)

    RF_RECO_predict_RL_2 = np.load(inputdata_path + 'RF_RL_RECO_predict_good_y.npy', allow_pickle=True)
    RF_RECO_true_RL_2 = np.load(inputdata_path + 'RF_RL_RECO_true_good_y.npy', allow_pickle=True)

    # ***************************************
    # Physics-aware RF三个输出 有气象数据的
    # XGB three  第一种情况 所有数据
    XGB_NEE_predict_RLM_0 = np.array(np.load(inputdata_path + 'XGB_RLM_NEE_predict_3.npy', allow_pickle=True), dtype=np.float64)
    XGB_NEE_true_RLM_0 = np.array(np.load(inputdata_path + 'XGB_RLM_NEE_true_3.npy', allow_pickle=True),
                                     dtype=np.float64)
    # XGB_NEE_true_RLM_0 = np.load(inputdata_path + 'XGB_RLM_NEE_true_3.npy', allow_pickle=True)
    # 计算并打印 XGB_NEE_predict_RLM_0 的统计信息

    XGB_GPP_predict_RLM_0 = np.load(inputdata_path + 'XGB_RLM_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true_RLM_0 = np.load(inputdata_path + 'XGB_RLM_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict_RLM_0 = np.load(inputdata_path + 'XGB_RLM_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true_RLM_0 = np.load(inputdata_path + 'XGB_RLM_RECO_true_3.npy', allow_pickle=True)

    XGB_NEE_predict_RL_0 = np.load(inputdata_path + 'XGB_RL_NEE_predict_3.npy', allow_pickle=True)
    XGB_NEE_true_RL_0 = np.load(inputdata_path + 'XGB_RL_NEE_true_3.npy', allow_pickle=True)

    XGB_GPP_predict_RL_0 = np.load(inputdata_path + 'XGB_RL_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true_RL_0 = np.load(inputdata_path + 'XGB_RL_GPP_true_3.npy', allow_pickle=True)

    XGB_RECO_predict_RL_0 = np.load(inputdata_path + 'XGB_RL_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true_RL_0 = np.load(inputdata_path + 'XGB_RL_RECO_true_3.npy', allow_pickle=True)

    # XGB three  第2种情况 只有质量好的modis数据
    XGB_NEE_predict_RLM_1 = np.load(inputdata_path + 'XGB_RLM_NEE_predict_good_modis.npy', allow_pickle=True)
    XGB_NEE_true_RLM_1 = np.load(inputdata_path + 'XGB_RLM_NEE_true_good_modis.npy', allow_pickle=True)

    XGB_GPP_predict_RLM_1 = np.load(inputdata_path + 'XGB_RLM_GPP_predict_good_modis.npy', allow_pickle=True)
    XGB_GPP_true_RLM_1 = np.load(inputdata_path + 'XGB_RLM_GPP_true_good_modis.npy', allow_pickle=True)

    XGB_RECO_predict_RLM_1 = np.load(inputdata_path + 'XGB_RLM_RECO_predict_good_modis.npy', allow_pickle=True)
    XGB_RECO_true_RLM_1 = np.load(inputdata_path + 'XGB_RLM_RECO_true_good_modis.npy', allow_pickle=True)

    XGB_NEE_predict_RL_1 = np.load(inputdata_path + 'XGB_RL_NEE_predict_good_modis.npy', allow_pickle=True)
    XGB_NEE_true_RL_1 = np.load(inputdata_path + 'XGB_RL_NEE_true_good_modis.npy', allow_pickle=True)

    XGB_GPP_predict_RL_1 = np.load(inputdata_path + 'XGB_RL_GPP_predict_good_modis.npy', allow_pickle=True)
    XGB_GPP_true_RL_1 = np.load(inputdata_path + 'XGB_RL_GPP_true_good_modis.npy', allow_pickle=True)

    XGB_RECO_predict_RL_1 = np.load(inputdata_path + 'XGB_RL_RECO_predict_good_modis.npy', allow_pickle=True)
    XGB_RECO_true_RL_1 = np.load(inputdata_path + 'XGB_RL_RECO_true_good_modis.npy', allow_pickle=True)

    # XGB three  第2种情况 只有质量好的modis数据
    XGB_NEE_predict_RLM_2 = np.load(inputdata_path + 'XGB_RLM_NEE_predict_good_y.npy', allow_pickle=True)
    XGB_NEE_true_RLM_2 = np.load(inputdata_path + 'XGB_RLM_NEE_true_good_y.npy', allow_pickle=True)

    XGB_GPP_predict_RLM_2 = np.load(inputdata_path + 'XGB_RLM_GPP_predict_good_y.npy', allow_pickle=True)
    XGB_GPP_true_RLM_2 = np.load(inputdata_path + 'XGB_RLM_GPP_true_good_y.npy', allow_pickle=True)

    XGB_RECO_predict_RLM_2 = np.load(inputdata_path + 'XGB_RLM_RECO_predict_good_y.npy', allow_pickle=True)
    XGB_RECO_true_RLM_2 = np.load(inputdata_path + 'XGB_RLM_RECO_true_good_y.npy', allow_pickle=True)

    XGB_NEE_predict_RL_2 = np.load(inputdata_path + 'XGB_RL_NEE_predict_good_y.npy', allow_pickle=True)
    XGB_NEE_true_RL_2 = np.load(inputdata_path + 'XGB_RL_NEE_true_good_y.npy', allow_pickle=True)

    XGB_GPP_predict_RL_2 = np.load(inputdata_path + 'XGB_RL_GPP_predict_good_y.npy', allow_pickle=True)
    XGB_GPP_true_RL_2 = np.load(inputdata_path + 'XGB_RL_GPP_true_good_y.npy', allow_pickle=True)

    XGB_RECO_predict_RL_2 = np.load(inputdata_path + 'XGB_RL_RECO_predict_good_y.npy', allow_pickle=True)
    XGB_RECO_true_RL_2 = np.load(inputdata_path + 'XGB_RL_RECO_true_good_y.npy', allow_pickle=True)

    # 计算各模型对应的RMSE; CC; Absolute bias;
    # 使用气象数据计算的值 深度学习
    DL_NEE_0_RLM_rmse, DL_NEE_0_RLM_cc, DL_NEE_0_RLM_bias = calculate_metrics(Ground_NEE_0_RLM_filtered,DL_NEE_0_RLM_filtered)
    DL_NEE_1_RLM_rmse, DL_NEE_1_RLM_cc, DL_NEE_1_RLM_bias = calculate_metrics(Ground_NEE_1_RLM_filtered,DL_NEE_1_RLM_filtered)
    DL_NEE_2_RLM_rmse, DL_NEE_2_RLM_cc, DL_NEE_2_RLM_bias = calculate_metrics(Ground_NEE_2_RLM_filtered,DL_NEE_2_RLM_filtered)

    DL_GPP_0_RLM_rmse, DL_GPP_0_RLM_cc, DL_GPP_0_RLM_bias = calculate_metrics(Ground_GPP_0_RLM_filtered,
                                                                              DL_GPP_0_RLM_filtered)
    DL_GPP_1_RLM_rmse, DL_GPP_1_RLM_cc, DL_GPP_1_RLM_bias = calculate_metrics(Ground_GPP_0_RLM_filtered,
                                                                              DL_GPP_0_RLM_filtered)
    DL_GPP_2_RLM_rmse, DL_GPP_2_RLM_cc, DL_GPP_2_RLM_bias = calculate_metrics(Ground_GPP_2_RLM_filtered,
                                                                              DL_GPP_2_RLM_filtered)
    DL_RECO_0_RLM_rmse, DL_RECO_0_RLM_cc, DL_RECO_0_RLM_bias = calculate_metrics(Ground_RECO_0_RLM_filtered,
                                                                              DL_RECO_0_RLM_filtered)
    DL_RECO_1_RLM_rmse, DL_RECO_1_RLM_cc, DL_RECO_1_RLM_bias = calculate_metrics(Ground_RECO_1_RLM_filtered,
                                                                              DL_RECO_1_RLM_filtered)
    DL_RECO_2_RLM_rmse, DL_RECO_2_RLM_cc, DL_RECO_2_RLM_bias = calculate_metrics(Ground_RECO_2_RLM_filtered,
                                                                              DL_RECO_2_RLM_filtered)

    # XGBOOST
    XGB_NEE_0_RLM_rmse, XGB_NEE_0_RLM_cc, XGB_NEE_0_RLM_bias = calculate_metrics(np.array(XGB_NEE_predict_RLM_0, dtype=np.float64),
                                                                                 np.array(XGB_NEE_true_RLM_0,
                                                                                          dtype=np.float64))

    XGB_NEE_1_RLM_rmse, XGB_NEE_1_RLM_cc, XGB_NEE_1_RLM_bias = calculate_metrics(np.array(XGB_NEE_predict_RLM_1, dtype=np.float64),
                                                                                 np.array(XGB_NEE_true_RLM_1,
                                                                                          dtype=np.float64))
    XGB_NEE_2_RLM_rmse, XGB_NEE_2_RLM_cc, XGB_NEE_2_RLM_bias = calculate_metrics(np.array(XGB_NEE_predict_RLM_2, dtype=np.float64),
                                                                                 np.array(XGB_NEE_true_RLM_2,
                                                                                          dtype=np.float64))
    XGB_GPP_0_RLM_rmse, XGB_GPP_0_RLM_cc, XGB_GPP_0_RLM_bias = calculate_metrics(np.array(XGB_GPP_predict_RLM_0, dtype=np.float64),
                                                                                 np.array(XGB_GPP_true_RLM_0,dtype=np.float64))
    XGB_GPP_1_RLM_rmse, XGB_GPP_1_RLM_cc, XGB_GPP_1_RLM_bias = calculate_metrics(np.array(XGB_GPP_predict_RLM_1, dtype=np.float64),
                                                                                 np.array(XGB_GPP_true_RLM_1,dtype=np.float64))
    XGB_GPP_2_RLM_rmse, XGB_GPP_2_RLM_cc, XGB_GPP_2_RLM_bias = calculate_metrics(np.array(XGB_GPP_predict_RLM_2, dtype=np.float64),
                                                                                 np.array(XGB_GPP_true_RLM_2,dtype=np.float64))
    XGB_RECO_0_RLM_rmse, XGB_RECO_0_RLM_cc, XGB_RECO_0_RLM_bias = calculate_metrics(np.array(XGB_RECO_predict_RLM_0, dtype=np.float64),
                                                                                 np.array(XGB_RECO_true_RLM_0,dtype=np.float64))
    XGB_RECO_1_RLM_rmse, XGB_RECO_1_RLM_cc, XGB_RECO_1_RLM_bias = calculate_metrics(np.array(XGB_RECO_predict_RLM_1, dtype=np.float64),
                                                                                 np.array(XGB_RECO_true_RLM_1,dtype=np.float64))
    XGB_RECO_2_RLM_rmse, XGB_RECO_2_RLM_cc, XGB_RECO_2_RLM_bias = calculate_metrics(np.array(XGB_RECO_predict_RLM_2, dtype=np.float64),
                                                                                 np.array(XGB_RECO_true_RLM_2,dtype=np.float64))
# 随机森林
# XGBOOST
    RF_NEE_0_RLM_rmse, RF_NEE_0_RLM_cc, RF_NEE_0_RLM_bias = calculate_metrics(
        np.array(RF_NEE_predict_RLM_0, dtype=np.float64),
        np.array(RF_NEE_true_RLM_0,
                 dtype=np.float64))

    RF_NEE_1_RLM_rmse, RF_NEE_1_RLM_cc, RF_NEE_1_RLM_bias = calculate_metrics(
        np.array(RF_NEE_predict_RLM_1, dtype=np.float64),
        np.array(RF_NEE_true_RLM_1,
                 dtype=np.float64))
    RF_NEE_2_RLM_rmse, RF_NEE_2_RLM_cc, RF_NEE_2_RLM_bias = calculate_metrics(
        np.array(RF_NEE_predict_RLM_2, dtype=np.float64),
        np.array(RF_NEE_true_RLM_2,
                 dtype=np.float64))
    RF_GPP_0_RLM_rmse, RF_GPP_0_RLM_cc, RF_GPP_0_RLM_bias = calculate_metrics(
        np.array(RF_GPP_predict_RLM_0, dtype=np.float64),
        np.array(RF_GPP_true_RLM_0, dtype=np.float64))
    RF_GPP_1_RLM_rmse, RF_GPP_1_RLM_cc, RF_GPP_1_RLM_bias = calculate_metrics(
        np.array(RF_GPP_predict_RLM_1, dtype=np.float64),
        np.array(RF_GPP_true_RLM_1, dtype=np.float64))
    RF_GPP_2_RLM_rmse, RF_GPP_2_RLM_cc, RF_GPP_2_RLM_bias = calculate_metrics(
        np.array(RF_GPP_predict_RLM_2, dtype=np.float64),
        np.array(RF_GPP_true_RLM_2, dtype=np.float64))
    RF_RECO_0_RLM_rmse, RF_RECO_0_RLM_cc, RF_RECO_0_RLM_bias = calculate_metrics(
        np.array(RF_RECO_predict_RLM_0, dtype=np.float64),
        np.array(RF_RECO_true_RLM_0, dtype=np.float64))
    RF_RECO_1_RLM_rmse, RF_RECO_1_RLM_cc, RF_RECO_1_RLM_bias = calculate_metrics(
        np.array(RF_RECO_predict_RLM_1, dtype=np.float64),
        np.array(RF_RECO_true_RLM_1, dtype=np.float64))
    RF_RECO_2_RLM_rmse, RF_RECO_2_RLM_cc, RF_RECO_2_RLM_bias = calculate_metrics(
        np.array(RF_RECO_predict_RLM_2, dtype=np.float64),
        np.array(XGB_RECO_true_RLM_2, dtype=np.float64))

    # **********************************************
    # 不使用气象数据计算的值 深度学习
    DL_NEE_0_RL_rmse, DL_NEE_0_RL_cc, DL_NEE_0_RL_bias = calculate_metrics(Ground_NEE_0_RL_filtered,
                                                                              DL_NEE_0_RL_filtered)
    DL_NEE_1_RL_rmse, DL_NEE_1_RL_cc, DL_NEE_1_RL_bias = calculate_metrics(Ground_NEE_1_RL_filtered,
                                                                              DL_NEE_1_RL_filtered)
    DL_NEE_2_RL_rmse, DL_NEE_2_RL_cc, DL_NEE_2_RL_bias = calculate_metrics(Ground_NEE_2_RL_filtered,
                                                                              DL_NEE_2_RL_filtered)

    DL_GPP_0_RL_rmse, DL_GPP_0_RL_cc, DL_GPP_0_RL_bias = calculate_metrics(Ground_GPP_0_RL_filtered,
                                                                              DL_GPP_0_RL_filtered)
    DL_GPP_1_RL_rmse, DL_GPP_1_RL_cc, DL_GPP_1_RL_bias = calculate_metrics(Ground_GPP_0_RL_filtered,
                                                                              DL_GPP_0_RL_filtered)
    DL_GPP_2_RL_rmse, DL_GPP_2_RL_cc, DL_GPP_2_RL_bias = calculate_metrics(Ground_GPP_2_RL_filtered,
                                                                              DL_GPP_2_RL_filtered)
    DL_RECO_0_RL_rmse, DL_RECO_0_RL_cc, DL_RECO_0_RL_bias = calculate_metrics(Ground_RECO_0_RL_filtered,
                                                                                 DL_RECO_0_RL_filtered)
    DL_RECO_1_RL_rmse, DL_RECO_1_RL_cc, DL_RECO_1_RL_bias = calculate_metrics(Ground_RECO_1_RL_filtered,
                                                                                 DL_RECO_1_RL_filtered)
    DL_RECO_2_RL_rmse, DL_RECO_2_RL_cc, DL_RECO_2_RL_bias = calculate_metrics(Ground_RECO_2_RL_filtered,
                                                                                 DL_RECO_2_RL_filtered)

    # XGBOOST
    XGB_NEE_0_RL_rmse, XGB_NEE_0_RL_cc, XGB_NEE_0_RL_bias = calculate_metrics(
        np.array(XGB_NEE_predict_RL_0, dtype=np.float64),
        np.array(XGB_NEE_true_RL_0,
                 dtype=np.float64))

    XGB_NEE_1_RL_rmse, XGB_NEE_1_RL_cc, XGB_NEE_1_RL_bias = calculate_metrics(
        np.array(XGB_NEE_predict_RL_1, dtype=np.float64),
        np.array(XGB_NEE_true_RL_1,
                 dtype=np.float64))
    XGB_NEE_2_RL_rmse, XGB_NEE_2_RL_cc, XGB_NEE_2_RL_bias = calculate_metrics(
        np.array(XGB_NEE_predict_RL_2, dtype=np.float64),
        np.array(XGB_NEE_true_RL_2,
                 dtype=np.float64))
    XGB_GPP_0_RL_rmse, XGB_GPP_0_RL_cc, XGB_GPP_0_RL_bias = calculate_metrics(
        np.array(XGB_GPP_predict_RL_0, dtype=np.float64),
        np.array(XGB_GPP_true_RL_0, dtype=np.float64))
    XGB_GPP_1_RL_rmse, XGB_GPP_1_RL_cc, XGB_GPP_1_RL_bias = calculate_metrics(
        np.array(XGB_GPP_predict_RL_1, dtype=np.float64),
        np.array(XGB_GPP_true_RL_1, dtype=np.float64))
    XGB_GPP_2_RL_rmse, XGB_GPP_2_RL_cc, XGB_GPP_2_RL_bias = calculate_metrics(
        np.array(XGB_GPP_predict_RL_2, dtype=np.float64),
        np.array(XGB_GPP_true_RL_2, dtype=np.float64))
    XGB_RECO_0_RL_rmse, XGB_RECO_0_RL_cc, XGB_RECO_0_RL_bias = calculate_metrics(
        np.array(XGB_RECO_predict_RL_0, dtype=np.float64),
        np.array(XGB_RECO_true_RL_0, dtype=np.float64))
    XGB_RECO_1_RL_rmse, XGB_RECO_1_RL_cc, XGB_RECO_1_RL_bias = calculate_metrics(
        np.array(XGB_RECO_predict_RL_1, dtype=np.float64),
        np.array(XGB_RECO_true_RL_1, dtype=np.float64))
    XGB_RECO_2_RL_rmse, XGB_RECO_2_RL_cc, XGB_RECO_2_RL_bias = calculate_metrics(
        np.array(XGB_RECO_predict_RL_2, dtype=np.float64),
        np.array(XGB_RECO_true_RL_2, dtype=np.float64))
    # 随机森林
    # XGBOOST
    RF_NEE_0_RL_rmse, RF_NEE_0_RL_cc, RF_NEE_0_RL_bias = calculate_metrics(
        np.array(RF_NEE_predict_RL_0, dtype=np.float64),
        np.array(RF_NEE_true_RL_0,
                 dtype=np.float64))

    RF_NEE_1_RL_rmse, RF_NEE_1_RL_cc, RF_NEE_1_RL_bias = calculate_metrics(
        np.array(RF_NEE_predict_RL_1, dtype=np.float64),
        np.array(RF_NEE_true_RL_1,
                 dtype=np.float64))
    RF_NEE_2_RL_rmse, RF_NEE_2_RL_cc, RF_NEE_2_RL_bias = calculate_metrics(
        np.array(RF_NEE_predict_RL_2, dtype=np.float64),
        np.array(RF_NEE_true_RL_2,
                 dtype=np.float64))
    RF_GPP_0_RL_rmse, RF_GPP_0_RL_cc, RF_GPP_0_RL_bias = calculate_metrics(
        np.array(RF_GPP_predict_RL_0, dtype=np.float64),
        np.array(RF_GPP_true_RL_0, dtype=np.float64))
    RF_GPP_1_RL_rmse, RF_GPP_1_RL_cc, RF_GPP_1_RL_bias = calculate_metrics(
        np.array(RF_GPP_predict_RL_1, dtype=np.float64),
        np.array(RF_GPP_true_RL_1, dtype=np.float64))
    RF_GPP_2_RL_rmse, RF_GPP_2_RL_cc, RF_GPP_2_RL_bias = calculate_metrics(
        np.array(RF_GPP_predict_RL_2, dtype=np.float64),
        np.array(RF_GPP_true_RL_2, dtype=np.float64))
    RF_RECO_0_RL_rmse, RF_RECO_0_RL_cc, RF_RECO_0_RL_bias = calculate_metrics(
        np.array(RF_RECO_predict_RL_0, dtype=np.float64),
        np.array(RF_RECO_true_RL_0, dtype=np.float64))
    RF_RECO_1_RL_rmse, RF_RECO_1_RL_cc, RF_RECO_1_RL_bias = calculate_metrics(
        np.array(RF_RECO_predict_RL_1, dtype=np.float64),
        np.array(RF_RECO_true_RL_1, dtype=np.float64))
    RF_RECO_2_RL_rmse, RF_RECO_2_RL_cc, RF_RECO_2_RL_bias = calculate_metrics(
        np.array(RF_RECO_predict_RL_2, dtype=np.float64),
        np.array(XGB_RECO_true_RL_2, dtype=np.float64))

    rmse_data_RLM = {
        ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RLM_rmse, DL_NEE_1_RLM_rmse, DL_NEE_2_RLM_rmse],
        ('NEE', 'XGBoost'): [XGB_NEE_0_RLM_rmse, XGB_NEE_1_RLM_rmse, XGB_NEE_2_RLM_rmse],
        ('NEE', 'RF'): [RF_NEE_0_RLM_rmse, RF_NEE_1_RLM_rmse, RF_NEE_2_RLM_rmse],
        ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RLM_rmse, DL_GPP_1_RLM_rmse, DL_GPP_2_RLM_rmse],
        ('GPP', 'XGBoost'): [XGB_GPP_0_RLM_rmse, XGB_GPP_1_RLM_rmse, XGB_GPP_2_RLM_rmse],
        ('GPP', 'RF'): [RF_GPP_0_RLM_rmse, RF_GPP_1_RLM_rmse, RF_GPP_2_RLM_rmse],
        ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RLM_rmse, DL_RECO_1_RLM_rmse, DL_RECO_2_RLM_rmse],
        ('RECO', 'XGBoost'): [XGB_RECO_0_RLM_rmse, XGB_RECO_1_RLM_rmse, XGB_RECO_2_RLM_rmse],
        ('RECO', 'RF'): [RF_RECO_0_RLM_rmse, RF_RECO_1_RLM_rmse, RF_RECO_2_RLM_rmse]
    }

    CC_data_RLM = {
        ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RLM_cc, DL_NEE_1_RLM_cc, DL_NEE_2_RLM_cc],
        ('NEE', 'XGBoost'): [XGB_NEE_0_RLM_cc, XGB_NEE_1_RLM_cc, XGB_NEE_2_RLM_cc],
        ('NEE', 'RF'): [RF_NEE_0_RLM_cc, RF_NEE_1_RLM_cc, RF_NEE_2_RLM_cc],
        ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RLM_cc, DL_GPP_1_RLM_cc, DL_GPP_2_RLM_cc],
        ('GPP', 'XGBoost'): [XGB_GPP_0_RLM_cc, XGB_GPP_1_RLM_cc, XGB_GPP_2_RLM_cc],
        ('GPP', 'RF'): [RF_GPP_0_RLM_cc, RF_GPP_1_RLM_cc, RF_GPP_2_RLM_cc],
        ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RLM_cc, DL_RECO_1_RLM_cc, DL_RECO_2_RLM_cc],
        ('RECO', 'XGBoost'): [XGB_RECO_0_RLM_cc, XGB_RECO_1_RLM_cc, XGB_RECO_2_RLM_cc],
        ('RECO', 'RF'): [RF_RECO_0_RLM_cc, RF_RECO_1_RLM_cc, RF_RECO_2_RLM_cc]
    }
    bias_data_RLM = {
            ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RLM_bias, DL_NEE_1_RLM_bias, DL_NEE_2_RLM_bias],
            ('NEE', 'XGBoost'): [XGB_NEE_0_RLM_bias, XGB_NEE_1_RLM_bias, XGB_NEE_2_RLM_bias],
            ('NEE', 'RF'): [RF_NEE_0_RLM_bias, RF_NEE_1_RLM_bias, RF_NEE_2_RLM_bias],
            ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RLM_bias, DL_GPP_1_RLM_bias, DL_GPP_2_RLM_bias],
            ('GPP', 'XGBoost'): [XGB_GPP_0_RLM_bias, XGB_GPP_1_RLM_bias, XGB_GPP_2_RLM_bias],
            ('GPP', 'RF'): [RF_GPP_0_RLM_bias, RF_GPP_1_RLM_bias, RF_GPP_2_RLM_bias],
            ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RLM_bias, DL_RECO_1_RLM_bias, DL_RECO_2_RLM_bias],
            ('RECO', 'XGBoost'): [XGB_RECO_0_RLM_bias, XGB_RECO_1_RLM_bias, XGB_RECO_2_RLM_bias],
            ('RECO', 'RF'): [RF_RECO_0_RLM_bias, RF_RECO_1_RLM_bias, RF_RECO_2_RLM_bias]
        }

    rmse_data_RL = {
        ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RL_rmse, DL_NEE_1_RL_rmse, DL_NEE_2_RL_rmse],
        ('NEE', 'XGBoost'): [XGB_NEE_0_RL_rmse, XGB_NEE_1_RL_rmse, XGB_NEE_2_RL_rmse],
        ('NEE', 'RF'): [RF_NEE_0_RL_rmse, RF_NEE_1_RL_rmse, RF_NEE_2_RL_rmse],
        ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RL_rmse, DL_GPP_1_RL_rmse, DL_GPP_2_RL_rmse],
        ('GPP', 'XGBoost'): [XGB_GPP_0_RL_rmse, XGB_GPP_1_RL_rmse, XGB_GPP_2_RL_rmse],
        ('GPP', 'RF'): [RF_GPP_0_RL_rmse, RF_GPP_1_RL_rmse, RF_GPP_2_RL_rmse],
        ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RL_rmse, DL_RECO_1_RL_rmse, DL_RECO_2_RL_rmse],
        ('RECO', 'XGBoost'): [XGB_RECO_0_RL_rmse, XGB_RECO_1_RL_rmse, XGB_RECO_2_RL_rmse],
        ('RECO', 'RF'): [RF_RECO_0_RL_rmse, RF_RECO_1_RL_rmse, RF_RECO_2_RL_rmse]
    }

    CC_data_RL = {
        ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RL_cc, DL_NEE_1_RL_cc, DL_NEE_2_RL_cc],
        ('NEE', 'XGBoost'): [XGB_NEE_0_RL_cc, XGB_NEE_1_RL_cc, XGB_NEE_2_RL_cc],
        ('NEE', 'RF'): [RF_NEE_0_RL_cc, RF_NEE_1_RL_cc, RF_NEE_2_RL_cc],
        ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RL_cc, DL_GPP_1_RL_cc, DL_GPP_2_RL_cc],
        ('GPP', 'XGBoost'): [XGB_GPP_0_RL_cc, XGB_GPP_1_RL_cc, XGB_GPP_2_RL_cc],
        ('GPP', 'RF'): [RF_GPP_0_RL_cc, RF_GPP_1_RL_cc, RF_GPP_2_RL_cc],
        ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RL_cc, DL_RECO_1_RL_cc, DL_RECO_2_RL_cc],
        ('RECO', 'XGBoost'): [XGB_RECO_0_RL_cc, XGB_RECO_1_RL_cc, XGB_RECO_2_RL_cc],
        ('RECO', 'RF'): [RF_RECO_0_RL_cc, RF_RECO_1_RL_cc, RF_RECO_2_RL_cc]
    }
    bias_data_RL = {
        ('NEE', 'Physics-aware transformer'): [DL_NEE_0_RL_bias, DL_NEE_1_RL_bias, DL_NEE_2_RL_bias],
        ('NEE', 'XGBoost'): [XGB_NEE_0_RL_bias, XGB_NEE_1_RL_bias, XGB_NEE_2_RL_bias],
        ('NEE', 'RF'): [RF_NEE_0_RL_bias, RF_NEE_1_RL_bias, RF_NEE_2_RL_bias],
        ('GPP', 'Physics-aware transformer'): [DL_GPP_0_RL_bias, DL_GPP_1_RL_bias, DL_GPP_2_RL_bias],
        ('GPP', 'XGBoost'): [XGB_GPP_0_RL_bias, XGB_GPP_1_RL_bias, XGB_GPP_2_RL_bias],
        ('GPP', 'RF'): [RF_GPP_0_RL_bias, RF_GPP_1_RL_bias, RF_GPP_2_RL_bias],
        ('RECO', 'Physics-aware transformer'): [DL_RECO_0_RL_bias, DL_RECO_1_RL_bias, DL_RECO_2_RL_bias],
        ('RECO', 'XGBoost'): [XGB_RECO_0_RL_bias, XGB_RECO_1_RL_bias, XGB_RECO_2_RL_bias],
        ('RECO', 'RF'): [RF_RECO_0_RL_bias, RF_RECO_1_RL_bias, RF_RECO_2_RL_bias]
    }
    # Prepare data
    variables = ['NEE', 'GPP', 'RECO']
    models = ['Physics-aware transformer', 'XGBoost', 'RF']
    Y_metrics = ['RMSE', 'CC', 'Absolute Bias']

    # RMSE, CC, and Bias data
    metrics_data = [rmse_data_RLM, CC_data_RLM, bias_data_RLM]
    # metrics_data = [rmse_data_RL, CC_data_RL, bias_data_RL]
    # Call function with data and plot
    bar_multi(metrics_data, variables, models, Y_metrics)
