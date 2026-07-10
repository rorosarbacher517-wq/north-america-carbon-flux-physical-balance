# 统计每个模型 每种植被功能类型对应的NEE GPP RECO
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

    X_variables = ['Ground_differ_0_RLM', 'Ground_differ_0_RLM', 'Ground_differ_0_RLM', 'Ground_differ_0_RLM',
                   'Ground_differ_0_RLM',
                   'Ground_differ_0', 'Ground_differ_0', 'Ground_differ_0', 'Ground_differ_0', 'Ground_differ_0']
    Y_variables = ['DL_differ_0_RLM', 'DL_differ_0_no_physical_RLM', 'DL_differ_0_RLM_Single', 'XGB_differ_0_RLM', 'RF_differ_0_RLM',
                   'DL_differ_0', 'DL_differ_0_no_physical', 'DL_differ_0_RL_Single', 'XGB_differ_0', 'RF_differ_0']
# *******************************************
# 获取每个站点对应的植被功能类型
    # csv文件中 读取Site_Id列和Vegetation列；获取站点名称和植被功能类型的对应关系，然后
    # Load your CSV file containing the site and vegetation data
    df_site_pft = pd.read_csv(
        "E:/The North America ecosystem carbon flux/Sites_estimation/data/Sites_csv/sites_reduced_HH_GE_shar_270.csv")

    # Extract the Site_Id and Vegetation columns
    site_id_to_pft = dict(zip(df_site_pft['Site_Id'].astype(str), df_site_pft['Vegetation'].astype(str)))

    DL_sites = DL_All_true[:, :, 0]  # Shape (1688, 365)
    # 初始化DL_PFT为全-9999（与DL_sites形状相同，字符串类型）
    DL_PFT = np.full_like(DL_All_true[:, :, 0], fill_value="-9999", dtype=object)

    # 获取所有唯一的站点ID字符串并遍历
    unique_sites = np.unique(DL_All_true[:, 0, 0])

    for site in unique_sites:
        if site in site_id_to_pft:
            mask = (DL_sites == site)
            DL_PFT[mask] = str(site_id_to_pft[site])  # 确保植被类型转为字符串
        else:
            print(f"Warning: Site {site} has no PFT mapping. Keeping -9999.")

    # Count valid entries
    valid_sites_count = np.sum(DL_All_true[:, :, 0:1] != "-9999")
    valid_pft_count = np.sum(DL_PFT != "-9999")

    # Print counts
    print(f'Count of valid sites (not -9999): {valid_sites_count}')
    print(f'Count of valid PFTs (not -9999): {valid_pft_count}')
    # Mock predictions for the 8 models corresponding to those in Y_variables
    predictions = {'DL_differ_0_RLM':DL_differ_0_RLM,
                   'DL_differ_0_no_physical_RLM':DL_differ_0_no_physical_RLM,
                   'DL_differ_0_RLM_Single':DL_differ_0_RLM_Single,
                   'XGB_differ_0_RLM':XGB_differ_0_RLM,
                    'RF_differ_0_RLM':RF_differ_0_RLM,
                    'DL_differ_0':DL_differ_0,
                    'DL_differ_0_no_physical':DL_differ_0_no_physical,
                    'DL_differ_0_RL_Single':DL_differ_0_RL_Single,
                    'XGB_differ_0':XGB_differ_0,
                    'RF_differ_0':RF_differ_0
    }

    # Flatten DL_PFT
    flat_DL_PFT = DL_PFT.flatten()  # Convert to 1D
    flat_DL_PFT = flat_DL_PFT[flat_DL_PFT != '-9999']  # Assuming you want to exclude '-9999'

    assert len(X_variables) == len(Y_variables), "X和Y变量数量必须一致"

    # Step 2: 创建模型预测字典（假设X_variables和Y_variables已对应）
    predictions = {
        model_name: model_pred  # 例如: 'DL_differ_0_RLM'对应DL_differ_0_RLM的预测值
        for model_name, model_pred in zip(Y_variables, [
            DL_differ_0_RLM, DL_differ_0_no_physical_RLM, DL_differ_0_RLM_Single,
            XGB_differ_0_RLM, RF_differ_0_RLM, DL_differ_0, DL_differ_0_no_physical,
            DL_differ_0_RL_Single, XGB_differ_0, RF_differ_0
        ])
    }

    # Step 3: 获取真实值（Ground Truth）对应的数据
    ground_truths = {
        y_var: globals()[x_var]  # 假设X_variables中的变量名对应全局变量中的真实值
        for x_var, y_var in zip(X_variables, Y_variables)
    }

    # Step 4: 按植被类型计算RMSE
    rmse_results = {vegetation_type: [] for vegetation_type in np.unique(flat_DL_PFT) if vegetation_type != "-9999"}

    for model_name in Y_variables:
        # 获取当前模型的预测值和对应的真实值
        pred = predictions[model_name]
        y_true = ground_truths[model_name]  # 修正：使用X_variable对应的真实值

        # 展平并过滤无效值
        flat_pred = pred.flatten()[flat_DL_PFT != '-9999']
        flat_y_true = y_true.flatten()[flat_DL_PFT != '-9999']

        # 分植被类型计算RMSE
        for vegetation_type in rmse_results.keys():
            mask = (flat_DL_PFT == vegetation_type)
            if np.sum(mask) > 0:
                rmse = np.sqrt(mean_squared_error(flat_y_true[mask], flat_pred[mask]))
                rmse_results[vegetation_type].append(rmse)
            else:
                rmse_results[vegetation_type].append(np.nan)

    # Step 5: 生成结果表格
    rmse_df = pd.DataFrame(rmse_results, index=Y_variables).T.reset_index()
    rmse_df.columns = ['Vegetation_Type'] + Y_variables
    print(rmse_df)
    # Step 1: 对RMSE结果四舍五入到两位小数
    rmse_df_rounded = rmse_df.round(2)  # 所有数值列保留两位小数

    # Step 2: 保存到CSV文件（路径可自定义）
    output_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/differ_rmse_by_PFT.csv"
    rmse_df_rounded.to_csv(output_path, index=False)

    # 打印保存路径确认
    print(f"RMSE结果已保存至：{output_path}")