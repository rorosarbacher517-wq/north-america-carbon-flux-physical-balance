# 统计每个模型 不同时间内 对应的NEE GPP RECO
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
import matplotlib.colors as mcolors

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
    DL_y_qa_mask
    DL_y_qa[:, :, 3]
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

    # DL_Y_predict_RLM = np.load(inputdata_path + 'DL_RLM_predict_good_modis.npy', allow_pickle=True)
    # DL_Y_true_RLM = np.load(inputdata_path + 'DL_RLM_true_good_modis.npy', allow_pickle=True)
    # DL_All_true_RLM = np.load(inputdata_path + 'DL_RLM_sites_good_modis.npy', allow_pickle=True)
    # inter_mask_RLM = np.load(inputdata_path + 'DL_RLM_x_mask_good_modis.npy', allow_pickle=True)
    # DL_y_qa_RLM = np.load(inputdata_path + 'DL_RLM_y_qa_good_modis.npy', allow_pickle=True)

    # DL_Y_predict_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_predict_no_physical_3.npy', allow_pickle=True)
    # DL_Y_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_true_no_physical_3.npy', allow_pickle=True)
    # DL_All_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_sites_no_physical_3.npy', allow_pickle=True)
    # inter_mask_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_x_mask_no_physical_3.npy', allow_pickle=True)
    # DL_y_qa_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_y_qa_no_physical_3.npy', allow_pickle=True)
    # 采用以下版本
    DL_Y_predict_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_predict_three_2.npy', allow_pickle=True)
    DL_Y_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_true_three_2.npy', allow_pickle=True)
    DL_All_true_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_sites_three_2.npy', allow_pickle=True)
    inter_mask_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_x_mask_three_2.npy', allow_pickle=True)
    DL_y_qa_no_physical_RLM = np.load(inputdata_path + 'DL_RLM_y_qa_three_2.npy', allow_pickle=True)

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

   #*******************************
    # DL 单独预测的 有气象数据
    DL_NEE_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_NEE_Single_2.npy', allow_pickle=True)
    DL_NEE_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_NEE_Single_2.npy', allow_pickle=True)
    #
    DL_GPP_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_GPP_Single_2.npy', allow_pickle=True)
    DL_GPP_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_GPP_Single_2.npy', allow_pickle=True)

    DL_RECO_predict_RLM_Single = np.load(inputdata_path + 'DL_RLM_predict_RECO_Single_3.npy', allow_pickle=True)
    DL_RECO_true_RLM_Single = np.load(inputdata_path + 'DL_RLM_true_RECO_Single_3.npy', allow_pickle=True)

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

    #*******************
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
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,is_inter=0)
    DL_NEE_1_no_physical_RLM, DL_GPP_1_no_physical_RLM, DL_RECO_1_no_physical_RLM, Ground_NEE_1_RLM, Ground_GPP_1_RLM, Ground_RECO_1_RLM, DL_NEE_1_RL_Single, DL_GPP_1_RL_Single, DL_RECO_1_RL_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,is_inter=1)
    DL_NEE_2_no_physical_RLM, DL_GPP_2_no_physical_RLM, DL_RECO_2_no_physical_RLM, Ground_NEE_2_RLM, Ground_GPP_2_RLM, Ground_RECO_2_RLM, DL_NEE_2_RL_Single, DL_GPP_2_RL_Single, DL_RECO_2_RL_Single = interpolation_quality(
        DL_Y_predict_no_physical_RLM, DL_Y_true_no_physical_RLM, inter_mask_no_physical_RLM, DL_y_qa_no_physical_RLM,
        DL_NEE_predict_RL_Single_filtered, DL_GPP_predict_RL_Single_filtered, DL_RECO_predict_RL_Single_filtered,is_inter=2)

    Ground_differ_0_RLM = Ground_NEE_0_RLM-Ground_RECO_0_RLM+Ground_GPP_0_RLM
    Ground_differ_0 = Ground_NEE_0- Ground_RECO_0 + Ground_GPP_0

    DL_differ_0_RLM = DL_NEE_0_RLM - DL_RECO_0_RLM + DL_GPP_0_RLM
    DL_differ_0 = DL_NEE_0- DL_RECO_0 + DL_GPP_0
    DL_differ_0_no_physical_RLM = DL_NEE_0_no_physical_RLM-  DL_RECO_0_no_physical_RLM + DL_GPP_0_no_physical_RLM
    DL_differ_0_no_physical = DL_NEE_0_no_physical - DL_RECO_0_no_physical + DL_GPP_0_no_physical
    DL_differ_0_RLM_Single = DL_NEE_0_RLM_Single - DL_RECO_0_RLM_Single + DL_GPP_0_RLM_Single
    DL_differ_0_RL_Single = DL_NEE_0_RL_Single - DL_RECO_0_RL_Single + DL_GPP_0_RL_Single
    XGB_differ_0_RLM = XGB_NEE_0_RLM - XGB_RECO_0_RLM + XGB_GPP_0_RLM
    XGB_differ_0 = XGB_NEE_0 - XGB_RECO_0 + XGB_GPP_0
    RF_differ_0_RLM = RF_NEE_0_RLM - RF_RECO_0_RLM + RF_GPP_0_RLM
    RF_differ_0 = RF_NEE_0 - RF_RECO_0 + RF_GPP_0

    # 定义模型名称
    model_names = [
        'XGBoost',
        'RF',
        'Transformer single-variable',
        'Transformer three-variables',
        'Physics-aware transformer',
    ]

    # # 定义模型与颜色的对应关系
    # color_mapping = {
    #     'Physics-aware transformer': 'red',
    #     'Transformer three-variables': 'orange',
    #     'Transformer single-variable': 'purple',
    #     'XGBoost': 'green',
    #     'RF': 'blue'
    # }
    # 定义模型与颜色的对应关系
    # color_mapping = {
    #     'XGBoost': 'green',  # 中海洋绿
    #     'RF': 'blue',  # 石灰蓝
    #     'Transformer single-variable': '#800080',  # 中紫色
    #     'Transformer three-variables': '#FFA500',  # 金色
    #     'Physics-aware transformer': 'red',  # 番茄红
    # }
    # # 定义模型与颜色的对应关系
    # color_mapping = {
    #     'Physics-aware transformer': '#FF6347',  # 番茄红
    #     'Transformer three-variables': '#FFA07A',  # 浅珊瑚色
    #     'Transformer single-variable': '#9370DB',  # 中紫色
    #     'XGBoost': '#3CB371',  # 中海洋绿
    #     'RF': '#4682B4'  # 石灰蓝
    # }
    # color_mapping = {
    #     'Physics-aware transformer': '#C72A2A',  # 深红色
    #     'Transformer three-variables': '#D88725',  # 深橙色
    #     'Transformer single-variable': '#59359E',  # 深紫色
    #     'XGBoost': '#2B8C68',  # 深海洋绿
    #     'RF': '#1F4E79'  # 深蓝色
    # }
    # 定义颜色
    colors1 = ['green', 'blue', 'purple', 'orange', 'red']  # Colors for the three models
    colors2 = ['#37AB78', '#8EC4D4', '#9370DB', '#FFA07A', '#F36B6F']  # Lighter colors

    # Combine colors1 and colors2
    colors = []
    for c1, c2 in zip(colors1, colors2):
        rgb1 = mcolors.to_rgb(c1)
        rgb2 = mcolors.to_rgb(c2)
        new_rgb = [(r1 + r2) / 2 for r1, r2 in zip(rgb1, rgb2)]
        colors.append(mcolors.to_hex(new_rgb))

    color_mapping = {
        'XGBoost': colors[0],  # 中海洋绿
        'RF': colors[1],  # 石灰蓝
        'Transformer single-variable': colors[2],  # 中紫色
        'Transformer three-variables': colors[3],  # 金色
        'Physics-aware transformer': colors[4],  # 番茄红
    }

    # 创建变量字典的函数
    def create_variable_definition(variable_type):
        return {
            'X_values': eval(f'Ground_{variable_type}_0_RLM.flatten()'),
            'Y_variables': [
                            f'XGB_{variable_type}_0_RLM',
                            f'RF_{variable_type}_0_RLM',
                            f'DL_{variable_type}_0_RLM_Single',
                            f'DL_{variable_type}_0_no_physical_RLM',
                            f'DL_{variable_type}_0_RLM',
                            f'XGB_{variable_type}_0',
                            f'RF_{variable_type}_0',
                            f'DL_{variable_type}_0_RL_Single',
                            f'DL_{variable_type}_0_no_physical',
                            f'DL_{variable_type}_0'],
            'Y_pred_values': {
                f'XGB_{variable_type}_0_RLM': eval(f'XGB_{variable_type}_0_RLM'),
                f'RF_{variable_type}_0_RLM': eval(f'RF_{variable_type}_0_RLM'),
                f'DL_{variable_type}_0_RLM_Single': eval(f'DL_{variable_type}_0_RLM_Single'),
                f'DL_{variable_type}_0_no_physical_RLM': eval(f'DL_{variable_type}_0_no_physical_RLM'),
                f'DL_{variable_type}_0_RLM': eval(f'DL_{variable_type}_0_RLM'),
                f'XGB_{variable_type}_0': eval(f'XGB_{variable_type}_0'),
                f'RF_{variable_type}_0': eval(f'RF_{variable_type}_0'),
                f'DL_{variable_type}_0_RL_Single': eval(f'DL_{variable_type}_0_RL_Single'),
                f'DL_{variable_type}_0_no_physical': eval(f'DL_{variable_type}_0_no_physical'),
                f'DL_{variable_type}_0': eval(f'DL_{variable_type}_0')

            },
            'Month_data': eval('DL_All_true[:, :, 5:6]'),  # 直接访问数据
            'Doy_data': eval('DL_All_true[:, :, 4:5]'),  # 直接访问数据
            'Annual_data': eval('DL_All_true[:, :, 3:4]')  # 直接访问数据
        }


    # 通过命名模型创建变量字典
    variables = {
        'NEE': create_variable_definition('NEE'),
        'GPP': create_variable_definition('GPP'),
        'RECO': create_variable_definition('RECO')
    }


    # 统计每天的相关系数并绘图
    def compute_daily_CC_and_plot(X_values, Y_pred_values, y_variables, doy_data, ax, title):
        flattened_DY = doy_data.flatten()
        valid_indices = flattened_DY != -9999
        flat_DY = flattened_DY[valid_indices]

        CC_results = []
        unique_doy = np.unique(flat_DY)

        for doy in unique_doy:
            mask = (flat_DY == doy)
            if np.sum(mask) == 0:
                continue  # 跳过没有数据的 DOY
            for i, var in enumerate(y_variables[:5]):  # 仅计算前5个模型
                if Y_pred_values[var].shape != X_values.shape:
                    raise ValueError(f"Shape mismatch for {var}: {Y_pred_values[var].shape} vs {X_values.shape}")
                CC = np.corrcoef(X_values[mask], Y_pred_values[var][mask])[0, 1]
                # CC = np.sqrt(np.mean((Y_pred_values[var][mask] - X_values[mask]) ** 2))
                CC_results.append({'DOY': doy, 'Model': model_names[i], 'CC': CC})

        df_CC = pd.DataFrame(CC_results)
        # 绘图，不显示图例
        palette = [color_mapping[model] for model in model_names]  # 应用颜色映射
        sns.lineplot(data=df_CC, x='DOY', y='CC', hue='Model', marker='o',
                     palette=palette, ax=ax, markersize=4, legend=False, markeredgecolor='none')

        # # 绘图，不显示图例
        # sns.lineplot(data=df_CC, x='DOY', y='CC', hue='Model', marker='o', palette="tab10", ax=ax, markersize=4,legend=False,markeredgecolor='none') #,legend=False
        ax.set_title(title, fontname='Arial', fontsize=18)

        ax.set_ylabel('CC', fontname='Arial', fontsize=18)
        ax.set_xticks(np.arange(0, 366, 30))  # 每30天显示一次
        ax.set_xticklabels(np.arange(0, 366, 30), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # 隐藏上面子图的横坐标
        # 隐藏上面子图的横坐标
        ax.set_xlabel('')
        # if ax != axes[0, 0]:  # 只保留最底部的横坐标
        #     ax.set_xlabel('')  # 隐藏横轴标签
        # else:
        #     ax.set_xlabel('DOY', fontsize=18)  # 设置字体大小
        ax.set_xlabel('DOY', fontsize=18)  # 设置字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)

    # 统计每月的相关系数并绘图
    def compute_monthly_CC_and_plot(X_values, Y_pred_values, y_variables, month_data, ax, title):
        flattened_MONTH = month_data.flatten()
        valid_indices = flattened_MONTH != -9999
        flat_MONTH = flattened_MONTH[valid_indices]
        # 定义每个月的天数（以365天为例）
        days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }

        CC_results = []
        unique_month = np.unique(flat_MONTH)

        for month in unique_month:
            month_days = days_in_month[month]
            mask = (flat_MONTH == month)
            if np.sum(mask) == 0:
                continue  # 跳过没有数据的 Month
            for i, var in enumerate(y_variables[:5]):  # 仅计算前5个模型
                if Y_pred_values[var].shape != X_values.shape:
                    raise ValueError(f"Shape mismatch for {var}: {Y_pred_values[var].shape} vs {X_values.shape}")
                X_values_month = X_values[mask] * month_days
                Y_pred_values_month = Y_pred_values[var][mask]* month_days
                CC = np.corrcoef(X_values_month, Y_pred_values_month)[0, 1]
                # CC = np.sqrt(np.mean((Y_pred_values[var][mask] - X_values[mask]) ** 2))
                CC_results.append({'Month': month, 'Model': model_names[i], 'CC': CC})

        df_CC = pd.DataFrame(CC_results)
        # 绘图，不显示图例
        palette = [color_mapping[model] for model in model_names]  # 应用颜色映射
        sns.lineplot(data=df_CC, x='Month', y='CC', hue='Model', marker='o',
                     palette=palette, ax=ax, markersize=4, legend=False, markeredgecolor='none')
        # # 绘图，不显示图例
        # sns.lineplot(data=df_CC, x='Month', y='CC', hue='Model', marker='o', palette="tab10", ax=ax, markersize=5,
        #              legend=False, markeredgecolor='none')
        if ax == axes[0, 1]:
            ax.set_title(title, fontname='Arial', fontsize=18)
        ax.set_ylabel('CC', fontname='Arial', fontsize=18)
        ax.set_xticks(np.arange(1, 13))  # 显示每个月的标签
        ax.set_xticklabels(np.arange(1, 13), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # 隐藏上面子图的横坐标
        # 隐藏上面子图的横坐标
        ax.set_xlabel('')
        # if ax != axes[2, 1]:  # 只保留最底部的横坐标
        #     ax.set_xlabel('')  # 隐藏横轴标签
        # else:
        #     ax.set_xlabel('Month', fontsize=18)  # 设置字体大小
        ax.set_xlabel('Month', fontsize=18)  # 设置字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)


    # 统计年度的相关系数并绘图
    def compute_annual_CC_and_plot(X_values, Y_pred_values, y_variables, annual_data, ax, title):
        flattened_YEAR = annual_data.flatten()
        valid_indices = flattened_YEAR != -9999
        flat_YEAR = flattened_YEAR[valid_indices]

        CC_results = []
        unique_years = np.unique(flat_YEAR).astype(int)

        for year in unique_years:
            mask = (flat_YEAR == year)
            if np.sum(mask) == 0:
                continue
            for i, var in enumerate(y_variables[:5]):
                if Y_pred_values[var].shape != X_values.shape:
                    raise ValueError(f"Shape mismatch for {var}: {Y_pred_values[var].shape} vs {X_values.shape}")
                CC = np.corrcoef(X_values[mask], Y_pred_values[var][mask])[0, 1]
                # CC = np.sqrt(np.mean((Y_pred_values[var][mask] - X_values[mask]) ** 2))
                CC_results.append({'Year': year, 'Model': model_names[i], 'CC': CC})

        df_CC = pd.DataFrame(CC_results)
        # 绘图，不显示图例
        palette = [color_mapping[model] for model in model_names]  # 应用颜色映射
        sns.lineplot(data=df_CC, x='Year', y='CC', hue='Model', marker='o',
                     palette=palette, ax=ax, markersize=4, legend=False, markeredgecolor='none')
        # # 绘图，不显示图例
        # sns.lineplot(data=df_CC, x='Year', y='CC', hue='Model', marker='o', palette="tab10", ax=ax, markersize=5,
        #              legend=False,markeredgecolor='none')
        if ax != axes[0, 2]:  # 只保留最底部的横坐标
            ax.set_title('')  # 隐藏横轴标签
        else:
            ax.set_title(title, fontname='Arial', fontsize=18)  # 设置字体大小
        # if ax == axes[0, 2]:
        #     ax.set_title(title, fontname='Arial', fontsize=18)
        ax.set_ylabel('CC', fontname='Arial', fontsize=18)
        ax.set_xticks(unique_years[::2])  # 仅每隔一年显示
        ax.set_xticklabels(unique_years[::2], rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # 隐藏上面子图的横坐标
        # 隐藏上面子图的横坐标
        ax.set_xlabel('')
        # if ax != axes[2, 2]:  # 只保留最底部的横坐标
        #     ax.set_xlabel('')  # 隐藏横轴标签
        # else:
        #     ax.set_xlabel('Year', fontsize=18)  # 设置字体大小
        ax.set_xlabel('Year', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)


    # 创建合并图（9个子图，3行3列）
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # 遍历变量并绘制每日、月度和年度 CC
    for i, (key, value) in enumerate(variables.items()):
        # 计算并绘制每日 CC
        compute_daily_CC_and_plot(
            value['X_values'],
            value['Y_pred_values'],
            value['Y_variables'],
            value['Doy_data'],
            axes[0, i],  # 第一列
            f'{key}'
        )

        # 计算并绘制月度 CC
        compute_monthly_CC_and_plot(
            value['X_values'],
            value['Y_pred_values'],
            value['Y_variables'],
            value['Month_data'],
            axes[1, i],  # 第二列
            f'CC of different models for monthly {key}'
        )

        # 计算并绘制年度 CC
        compute_annual_CC_and_plot(
            value['X_values'],
            value['Y_pred_values'],
            value['Y_variables'],
            value['Annual_data'],
            axes[2, i],  # 第三列
            f'CC of different models for annual {key}'
        )

    # 预留底部空间，确保图例不被tight_layout覆盖
    # fig.subplots_adjust(bottom=0.05)

    # 添加全局图例
    # 定义颜色调色板
    palette = [color_mapping[model] for model in model_names]

    # 设置图例，`bbox_to_anchor`用于确定图例位置，`ncol`用于定义列数
    handles = [plt.Line2D([0], [0], linestyle='-', marker='o', color=palette[i], label=model) for i, model in enumerate(model_names)]

    fig.legend(handles=handles, loc='lower right',ncol=len(model_names), fontsize=18, columnspacing=1.8, markerscale=2)
    # # 设置图例，`bbox_to_anchor`用于确定图例位置，`ncol`用于定义列数
    # fig.legend(handles=handles, title='Models', loc='center', bbox_to_anchor=(0.5, -0.03),
    #            ncol=len(model_names), fontsize=16, columnspacing=4, markerscale=2)

    # 调整布局，避免图例被遮挡
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) #rect=[0, 0.08, 1, 0.95]

    results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/'
    # 保存与显示
    # plt.savefig(results_path + 'Combined CC of all models 0406.png', dpi=300, bbox_inches='tight')
    plt.show()
