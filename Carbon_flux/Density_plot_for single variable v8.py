# 检查soil 的影响情况
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

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
def plot_scatter_with_reduced_low_density(x, y, z, ax, threshold=1, reduction_factor=0.5):
    """
    绘制散点图，同时随机减少密度小于指定阈值的点的显示数量。

    :param x: x 数据数组
    :param y: y 数据数组
    :param z: 密度数组
    :param ax: matplotlib axes 对象
    :param threshold: 密度阈值
    :param reduction_factor: 减少的比例 (0.5表示减少50%)
    """
    # 绘制主对角线
    ax.plot([common_min, common_max], [common_min, common_max], color='#535CA8', linestyle='--', linewidth=1)

    # 找出低密度点并进行处理
    low_density_mask = z < threshold
    print(len(low_density_mask))
    x_low = x[low_density_mask]
    y_low = y[low_density_mask]
    z_low = z[low_density_mask]

    # print(len(x_low))
    # 计算要保留的低密度点数量
    if len(x_low) > 0:
        num_to_keep = int(len(x_low) * (1 - reduction_factor))
        if num_to_keep > 0:
            # 随机选择要保留的低密度点
            low_density_indices = np.random.choice(len(x_low), size=num_to_keep, replace=False)
            x_low = x_low[low_density_indices]
            y_low = y_low[low_density_indices]
            z_low = z_low[low_density_indices]

    # 找出高密度点
    high_density_mask = z >= threshold

    x_high = x[high_density_mask]
    y_high = y[high_density_mask]
    z_high = z[high_density_mask]
    print(len(x_low))
    # 更新 x, y, z，确保它们的长度一致
    x_combined = np.concatenate((x_high, x_low))
    y_combined = np.concatenate((y_high, y_low))
    z_combined = np.concatenate((z_high, z_low))

    # 绘制散点图
    scatter = ax.scatter(x_combined, y_combined, c=z_combined, cmap='Spectral_r', s=5, edgecolor='none')

    # 返回绘制的散点句柄以供进一步处理（如添加colorbar）
    return scatter

if __name__ == "__main__":
    # download the data
    inputdata_path = './data/sites_results/'
    # inputdata_path = './data/Sites_results_copy/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RLM_predict_soil_1.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RLM_true_soil_1.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RLM_sites_soil_1.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RLM_x_mask_soil_1.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RLM_y_qa_soil_1.npy', allow_pickle=True)
    # DL_Y_true[:,:,1]
    DL_Y_predict_no_physical = np.load(inputdata_path + 'DL_RLM_predict_soil_no_physical.npy', allow_pickle=True)
    DL_Y_true_no_physical = np.load(inputdata_path + 'DL_RLM_true_soil_no_physical.npy', allow_pickle=True)
    DL_All_true_no_physical = np.load(inputdata_path + 'DL_RLM_sites_soil_no_physical.npy', allow_pickle=True)
    inter_mask_no_physical = np.load(inputdata_path + 'DL_RLM_x_mask_soil_no_physical.npy', allow_pickle=True)
    DL_y_qa_no_physical = np.load(inputdata_path + 'DL_RLM_y_qa_soil_no_physical.npy', allow_pickle=True)

    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RL_NEE_predict_soil.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RL_NEE_true_soil.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RL_GPP_predict_soil.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RL_GPP_true_soil.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RL_RECO_predict_soil.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RL_RECO_true_soil.npy', allow_pickle=True)

    # XGB three
    XGB_NEE_predict = np.load(inputdata_path + 'XGB_RL_NEE_predict_soil.npy', allow_pickle=True)
    XGB_NEE_true = np.load(inputdata_path + 'XGB_RL_NEE_true_3.npy', allow_pickle=True)

    XGB_GPP_predict = np.load(inputdata_path + 'XGB_RL_GPP_predict_soil.npy', allow_pickle=True)
    XGB_GPP_true = np.load(inputdata_path + 'XGB_RL_GPP_true_soil.npy', allow_pickle=True)

    XGB_RECO_predict = np.load(inputdata_path + 'XGB_RL_RECO_predict_soil.npy', allow_pickle=True)
    XGB_RECO_true = np.load(inputdata_path + 'XGB_RL_RECO_true_soil.npy', allow_pickle=True)

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
    X_variables = ['Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0_RLM','Ground_NEE_0','Ground_NEE_0','Ground_NEE_0', 'Ground_NEE_0']
    Y_variables = ['DL_NEE_0_RLM','DL_NEE_0_no_physical_RLM', 'XGB_NEE_0_RLM', 'RF_NEE_0_RLM','DL_NEE_0','DL_NEE_0_no_physical','XGB_NEE_0', 'RF_NEE_0']
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

    # 设置字体
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.family'] = 'Arial'
    # 创建图形和子图
    fig, axs = plt.subplots(2, 4, figsize=(16, 9))

    # 找到x和y的总体最小值和最大值
    overall_x_min = min(globals()[var].min() for var in X_variables)
    overall_x_max = max(globals()[var].max() for var in X_variables)
    overall_y_min = min(globals()[var].min() for var in Y_variables)
    overall_y_max = max(globals()[var].max() for var in Y_variables)

    # 初始化公共坐标轴范围
    common_min = -3
    common_max = 3

    # 颜色定义
    colors = [
        'white', 'lightcyan', 'lightblue', 'deepskyblue', 'blue',
        'mediumseagreen', 'green', 'greenyellow', 'yellow', 'gold',
        'orange', 'tomato', 'red', 'firebrick', 'darkred'
    ]

    n_bins = len(colors)
    first_bounds = None  # 用于存储第一个子图的边界

    for i in range(8):
        row = i // 4
        col = i % 4
        x_variable = X_variables[i]
        y_variable = Y_variables[i]
        x = globals()[x_variable]
        y = globals()[y_variable]

        # 计算统计数据
        rmse = np.sqrt(mean_squared_error(x, y))
        cc = np.corrcoef(x, y)[0, 1]

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
        text = f"RMSE = {rmse:.2f}\nCC = {cc:.2f}"
        axs[row, col].text(0.05, 0.97, text, verticalalignment='top', transform=axs[row, col].transAxes,
                           color='red', fontweight='bold', fontsize=16)

        # 绘制对角线
        axs[row, col].plot([common_min, common_max], [common_min, common_max], color='#535CA8', linestyle='--',
                           linewidth=1)

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
    # # 创建 legend 的 handles 和 labels
    # handles = []
    # labels = []
    #
    # for color, label in zip(reversed(colors[4:]), reversed(bounds[4:])):  # 反转顺序
    #     handles.append(mpatches.Patch(color=color))  # 使用 Patch 创建颜色条
    #     labels.append(f"{label:.0f}")  # 格式化标签
    #
    # # 自定义图例的绘制位置和大小
    # legend_position = [0.9, 0.3, 0.015, 0.75]  # 自定义位置
    #
    # # 使用 fig.legend 将图例放置在自定义位置
    # fig.legend(handles, labels, title="Density Levels", title_fontsize='13', fontsize='12',
    #            loc='center left', frameon=False,
    #            bbox_to_anchor=legend_position[:2], bbox_transform=fig.transFigure)
    # 创建图例位置
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
    y_labels = [f"{bound:.0f}" for bound in bounds[5:]]   # 添加一个标签以包含最后的边界

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
    fig.text(0.5, 0.05, 'NEE (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center', fontsize=18,
             fontname='Arial')
    # fig.text(0.03, 0.7, 'Input with\nmeteorological data', ha='center', va='center', rotation='vertical',
    #          fontsize=18, fontname='Arial')
    # fig.text(0.03, 0.3, 'Input without\nmeteorological data', ha='center', va='center', rotation='vertical',
    #          fontsize=18, fontname='Arial')
    fig.text(0.03, 0.7, 'Input with\nsoil content', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Arial')
    fig.text(0.03, 0.3, 'Input without\nsoil content', ha='center', va='center', rotation='vertical',
             fontsize=18, fontname='Arial')
    # 调整整体布局
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.35, hspace=0.35)

    # base_name = 'Comparison of predicted NEE by the CFR model and machine learning models using RLFM(1) or RLF(2) as input'
    # plt.suptitle(base_name, ha='center', fontsize=12, y=0.99, fontname='Arial')

    resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(resluts_path + 'Fig 3_soil_0313' + '.png', dpi=300)
    # Display the plot
    plt.show()

