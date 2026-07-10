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
    inputdata_path = './data/sites_output/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RL_predict_1.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RL_true_1.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RL_sites_1.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RL_x_mask_1.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RL_y_qa_1.npy', allow_pickle=True)
    DL_y_qa[:,:,3]
    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RL_NEE_predict_1.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RL_NEE_true_1.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RL_GPP_predict_1.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RL_GPP_true_1.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RL_RECO_predict_1.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RL_RECO_true_1.npy', allow_pickle=True)

    # 第一组值:插值与不插值的一起
    DL_NEE_0, DL_GPP_0, DL_RECO_0, Ground_NEE_0, Ground_GPP_0, Ground_RECO_0,RF_NEE_0, RF_GPP_0, RF_RECO_0 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa,RF_NEE_predict,RF_GPP_predict,RF_RECO_predict, is_inter=0)
    DL_NEE_1, DL_GPP_1, DL_RECO_1, Ground_NEE_1, Ground_GPP_1, Ground_RECO_1,RF_NEE_1, RF_GPP_1, RF_RECO_1 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa,RF_NEE_predict,RF_GPP_predict,RF_RECO_predict, is_inter=1)
    DL_NEE_2, DL_GPP_2, DL_RECO_2, Ground_NEE_2, Ground_GPP_2, Ground_RECO_2,RF_NEE_2, RF_GPP_2, RF_RECO_2 = interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa,RF_NEE_predict,RF_GPP_predict,RF_RECO_predict, is_inter=2)
    X_variables = ['Ground_NEE_0', 'Ground_GPP_0', 'Ground_RECO_0', 'Ground_NEE_1', 'Ground_GPP_1', 'Ground_RECO_1',
                   'Ground_NEE_2', 'Ground_GPP_2', 'Ground_RECO_2']
    Y_variables = ['RF_NEE_0', 'RF_GPP_0', 'RF_RECO_0', 'RF_NEE_1', 'RF_GPP_1', 'RF_RECO_1', 'RF_NEE_2', 'RF_GPP_2',
                   'RF_RECO_2']
    X_labels = ['Ground NEE overall', 'Ground GPP overall', 'Ground RECO overall', 'Ground NEE with no-interpolation', 'Ground GPP with no-interpolation', 'Ground RECO with no-interpolation',
                   'Ground NEE with no-interpolation and y-quality', 'Ground GPP with no-interpolation and y-quality', 'Ground RECO with no-interpolation and y-quality']
    Y_labels = ['RF NEE overall', 'RF GPP overall', 'RF RECO overall', 'RF NEE with no-interpolation', 'RF GPP with no-interpolation', 'RF RECO with no-interpolation',
                   'RF NEE with no-interpolation and y-quality', 'RF GPP with no-interpolation and y-quality', 'RF RECO with no-interpolation and y-quality']
    ax_subtitle = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    # 创建一个2x3的子图，共6个散点密度子图
    # Set the global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(3, 3, figsize=(14, 14))
    # 绘制散点密度图并添加评估指标
    for i in range(9):
        row = i // 3
        col = i % 3
        x_variable = X_variables[i]
        y_variable = Y_variables[i]
        # Get the data for the current variable
        x = globals()[x_variable] # Get the data for the RF variable
        y = globals()[y_variable] # Get the data for the DL variable

        # Calculate point density
        xy = np.vstack([x, y]).astype(float)
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        # Normalize the density values to the range 0 to 1
        z = (z - np.min(z)) / (np.max(z) - np.min(z))
        # # Plot the scatter density plot and set the frequency (probability)
        scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
        plt.colorbar(scatter, ax=axs[row, col], label='Density')
        min_val = min(np.min(y), np.min(y))
        max_val = max(np.max(y), np.max(y))
        axs[row, col].set_xlim([min_val, max_val])
        axs[row, col].set_ylim([min_val, max_val])
        axs[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        # Calculate RMSE, R2, and bias evaluation metrics
        rmse = np.sqrt(mean_squared_error(x, y))
        r2 = r2_score(x, y)
        n = len(x)
        bias = np.mean(x - y)
        cc = np.corrcoef(x, y)[0, 1]
        # 计算相对均方根误差（RRMSE）
        mean_value = np.mean(abs(x))
        rrmse = rmse / mean_value
        # Add text annotations
        text = "RMSE = {:.2f}\nRRMSE = {:.2f}\nCC = {:.2f}\nBias = {:.2f}\nn = {}".format(rmse, rrmse, cc, bias, n)
        axs[row, col].text(0.05, 0.95, text, verticalalignment='top', transform=axs[row, col].transAxes,
                           bbox=dict(facecolor='none', edgecolor='red'), color='red', fontweight='bold', fontname='Times New Roman')
        axs[row, col].set_xlabel(X_labels[i], fontname='Times New Roman')
        axs[row, col].set_ylabel(Y_labels[i], fontname='Times New Roman')
        axs[row, col].set_title(f"{ax_subtitle[i]}", fontname='Times New Roman')
    base_name = 'Results of Random Forest using reflectances, LAI, and FPAR variables'
    # unit = r'$\mathrm{g\ C\ m^{-2}\ d^{-1}}$'
    unit = ' (g C m$^{-2}$ d$^{-1}$)'
    plt.suptitle(base_name + unit , ha='center', fontsize=14, y=0.99)
    plt.tight_layout()
    resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    plt.savefig(resluts_path+base_name +'2'+ '.png', dpi=600)
    plt.show()

