# 统计每个模型 的bias

from statistics import mean

import numpy as np
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
        'DL_NEE_0_RLM_Single':DL_NEE_0_RLM_Single,
        'XGB_NEE_0_RLM': XGB_NEE_0_RLM,
        'RF_NEE_0_RLM': RF_NEE_0_RLM,
        'DL_NEE_0': DL_NEE_0,
        'DL_NEE_0_no_physical': DL_NEE_0_no_physical,
        'DL_NEE_0_RL_Single': DL_NEE_0_RL_Single,
        'XGB_NEE_0': XGB_NEE_0,
        'RF_NEE_0': RF_NEE_0,

        'DL_GPP_0_RLM': DL_GPP_0_RLM,
        'DL_GPP_0_no_physical_RLM': DL_GPP_0_no_physical_RLM,
        'DL_GPP_0_RLM_Single': DL_GPP_0_RLM_Single,
        'XGB_GPP_0_RLM': XGB_GPP_0_RLM,
        'RF_GPP_0_RLM': RF_GPP_0_RLM,
        'DL_GPP_0': DL_GPP_0,
        'DL_GPP_0_no_physical': DL_GPP_0_no_physical,
        'DL_GPP_0_RL_Single': DL_GPP_0_RL_Single,
        'XGB_GPP_0': XGB_GPP_0,
        'RF_GPP_0': RF_GPP_0,

        'DL_RECO_0_RLM': DL_RECO_0_RLM,
        'DL_RECO_0_no_physical_RLM': DL_RECO_0_no_physical_RLM,
        'DL_RECO_0_RLM_Single': DL_RECO_0_RLM_Single,
        'XGB_RECO_0_RLM': XGB_RECO_0_RLM,
        'RF_RECO_0_RLM': RF_RECO_0_RLM,
        'DL_RECO_0': DL_RECO_0,
        'DL_RECO_0_no_physical': DL_RECO_0_no_physical,
        'DL_RECO_0_RL_Single': DL_RECO_0_RL_Single,
        'XGB_RECO_0': XGB_RECO_0,
        'RF_RECO_0': RF_RECO_0,
    }
    df_ground_truth = pd.DataFrame(ground_truth_data)
    df_predictions = pd.DataFrame(pred_data)
    biases_with_met = pd.DataFrame({
        'DL_NEE_RLM': df_predictions['DL_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical_RLM': df_predictions['DL_NEE_0_no_physical_RLM'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_0_RLM_Single':df_predictions['DL_NEE_0_RLM_Single'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE_RLM': df_predictions['XGB_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE_RLM': df_predictions['RF_NEE_0_RLM'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP_RLM': df_predictions['DL_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical_RLM': df_predictions['DL_GPP_0_no_physical_RLM'] - df_ground_truth['Ground_GPP_0'],
        'DL_GPP_0_RLM_Single': df_predictions['DL_GPP_0_RLM_Single'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP_RLM': df_predictions['XGB_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP_RLM': df_predictions['RF_GPP_0_RLM'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO_RLM': df_predictions['DL_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical_RLM': df_predictions['DL_RECO_0_no_physical_RLM'] - df_ground_truth['Ground_RECO_0'],
        'DL_RECO_0_RLM_Single': df_predictions['DL_RECO_0_RLM_Single'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO_RLM': df_predictions['XGB_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO_RLM': df_predictions['RF_RECO_0_RLM'] - df_ground_truth['Ground_RECO_0'],
    })

    biases_without_met = pd.DataFrame({
        'DL_NEE': df_predictions['DL_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'DL_NEE_no_physical': df_predictions['DL_NEE_0_no_physical'] - df_ground_truth['Ground_NEE_0'],
        'DL_NEE_0_RL_Single': df_predictions['DL_NEE_0_RL_Single'] - df_ground_truth['Ground_NEE_0'],
            'XGB_NEE': df_predictions['XGB_NEE_0'] - df_ground_truth['Ground_NEE_0'],
            'RF_NEE': df_predictions['RF_NEE_0'] - df_ground_truth['Ground_NEE_0'],

            'DL_GPP': df_predictions['DL_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_no_physical': df_predictions['DL_GPP_0_no_physical'] - df_ground_truth['Ground_GPP_0'],
            'DL_GPP_0_RL_Single': df_predictions['DL_GPP_0_RL_Single'] - df_ground_truth['Ground_GPP_0'],
            'XGB_GPP': df_predictions['XGB_GPP_0'] - df_ground_truth['Ground_GPP_0'],
            'RF_GPP': df_predictions['RF_GPP_0'] - df_ground_truth['Ground_GPP_0'],

            'DL_RECO': df_predictions['DL_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'DL_RECO_no_physical': df_predictions['DL_RECO_0_no_physical'] - df_ground_truth['Ground_RECO_0'],
        'DL_RECO_0_RL_Single': df_predictions['DL_RECO_0_RL_Single'] - df_ground_truth['Ground_RECO_0'],
            'XGB_RECO': df_predictions['XGB_RECO_0'] - df_ground_truth['Ground_RECO_0'],
            'RF_RECO': df_predictions['RF_RECO_0'] - df_ground_truth['Ground_RECO_0']
    })

    data = {
        'Ground NEE_RLM': df_ground_truth['Ground_NEE_0'],
        'DL_NEE_RLM': biases_with_met['DL_NEE_RLM'],
        'DL_NEE_no_physical_RLM': biases_with_met['DL_NEE_no_physical_RLM'],
        'DL_NEE_0_RLM_Single':biases_with_met['DL_NEE_0_RLM_Single'],
        'XGB_NEE_RLM': biases_with_met['XGB_NEE_RLM'],
        'RF_NEE_RLM': biases_with_met['RF_NEE_RLM'],

        'Ground GPP_RLM': df_ground_truth['Ground_GPP_0'],
        'DL_GPP_RLM': biases_with_met['DL_GPP_RLM'],
        'DL_GPP_no_physical_RLM': biases_with_met['DL_GPP_no_physical_RLM'],
        'DL_GPP_0_RLM_Single': biases_with_met['DL_GPP_0_RLM_Single'],
        'XGB_GPP_RLM': biases_with_met['XGB_GPP_RLM'],
        'RF_GPP_RLM': biases_with_met['RF_GPP_RLM'],

        'Ground RECO_RLM': df_ground_truth['Ground_RECO_0'],
        'DL_RECO_RLM': biases_with_met['DL_RECO_RLM'],
        'DL_RECO_no_physical_RLM': biases_with_met['DL_RECO_no_physical_RLM'],
        'DL_RECO_0_RLM_Single': biases_with_met['DL_RECO_0_RLM_Single'],
        'XGB_RECO_RLM': biases_with_met['XGB_RECO_RLM'],
        'RF_RECO_RLM': biases_with_met['RF_RECO_RLM'],

        'Ground NEE': df_ground_truth['Ground_NEE_0'],
        'DL_NEE': biases_without_met['DL_NEE'],
        'DL_NEE_no_physical': biases_without_met['DL_NEE_no_physical'],
        'DL_NEE_0_RL_Single': biases_without_met['DL_NEE_0_RL_Single'],
        'XGB_NEE': biases_without_met['XGB_NEE'],
        'RF_NEE': biases_without_met['RF_NEE'],

        'Ground GPP': df_ground_truth['Ground_GPP_0'],
        'DL_GPP': biases_without_met['DL_GPP'],
        'DL_GPP_no_physical': biases_without_met['DL_GPP_no_physical'],
        'DL_GPP_0_RL_Single': biases_without_met['DL_GPP_0_RL_Single'],
        'XGB_GPP': biases_without_met['XGB_GPP'],
        'RF_GPP': biases_without_met['RF_GPP'],

        'Ground RECO': df_ground_truth['Ground_RECO_0'],
        'DL_RECO': biases_without_met['DL_RECO'],
        'DL_RECO_no_physical': biases_without_met['DL_RECO_no_physical'],
        'DL_RECO_0_RL_Single': biases_without_met['DL_RECO_0_RL_Single'],
        'XGB_RECO': biases_without_met['XGB_RECO'],
        'RF_RECO': biases_without_met['RF_RECO'],
    }

    # 创建数据框
    # 创建数据框
    df = pd.DataFrame(data)
    plt.rcParams['font.family'] = 'Arial'
    # model_subtitle = ['Physics-aware\ntransformer', 'XGBoost', 'RF']
# 定义模型名称
    model_subtitle  = [
        'XGBoost',
        'RF',
        'Transformer single-variable',
        'Transformer three-variables',
        'Physics-aware transformer',
    ]
    # model_subtitle = ['DL_physic', 'DL_no_physic', 'DL_single']
    # 定义颜色
    colors1 = ['green','blue','purple','orange','red']  # Colors for the three models
    colors2 = ['#37AB78', '#8EC4D4','#9370DB', '#FFA07A','#F36B6F']  # Lighter colors

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

        # 动态计算最小值和最大值
        min_val = df_sorted[ground_col].min()
        max_val = df_sorted[ground_col].max()

        # 根据间隔生成分箱
        interval = 1
        bins = np.arange(min_val, max_val + interval, interval)
        df_sorted['Bins'] = pd.cut(df_sorted[ground_col], bins)

        # 计算每个分箱的平均值
        mean_values = df_sorted.groupby('Bins')[model_cols].mean()

        # 提取模型的偏差数据
        rf_col = model_cols[0]  # RF模型列
        xgb_col = model_cols[1]  # XGBoost模型列
        phys_col = model_cols[4]  # Physics-aware模型列

        # 计算每个bin的绝对偏差
        rf_bias = mean_values[rf_col].abs()
        xgb_bias = mean_values[xgb_col].abs()
        phys_bias = mean_values[phys_col].abs()

        # 计算偏差百分比
        with np.errstate(divide='ignore', invalid='ignore'):
            rf_percent_diff = np.where(rf_bias != 0, (rf_bias - phys_bias) / rf_bias * 100, np.nan)
            xgb_percent_diff = np.where(xgb_bias != 0, (xgb_bias - phys_bias) / xgb_bias * 100, np.nan)

        # 找到最大有效差异百分比
        max_rf_diff = np.nanmax(rf_percent_diff)
        max_xgb_diff = np.nanmax(xgb_percent_diff)
        max_diff = max(max_rf_diff, max_xgb_diff)

        # 判断最大差异来源
        if max_diff == max_rf_diff:
            max_model = "Physics-aware vs. RF"
        else:
            max_model = "Physics-aware vs. XGBoost"

        # 打印结果到控制台
        print(f"子图 {index + 1} [{title}] 最大改进百分比: {max_diff:.0f}%")

        # 在图中添加文本标注
        ax.text(0.97, 0.97, f"Max bias reduction: {max_diff:.0f}%\n({max_model})",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=16, bbox=dict(facecolor='white', alpha=0.8))

        # 只取右区间值
        right_edges = [bin.left for bin in mean_values.index]

        # 绘制模型数据
        for model, color in zip(model_cols, colors):
            ax.plot(right_edges, mean_values[model], marker='o', label=model, color=color)

        # 添加虚线
        ax.axhline(0, color='gray', linestyle='--')

        if index < 3:
            ax.set_title(title, fontsize=24)

        ax.set_xticks(right_edges)
        ax.set_xticklabels([f"{edge:.0f}" for edge in right_edges], rotation=0)
        ax.set_xlabel('Flux tower bins (g C m$^{-2}$ d$^{-1}$)', fontsize=23)
        ax.set_ylabel('Average Bias (g C m$^{-2}$ d$^{-1}$)', fontsize=23)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=20)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(20)

    # 创建子图
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    groups = [
        ('Ground NEE', ['XGB_NEE_RLM','RF_NEE_RLM','DL_NEE_0_RLM_Single','DL_NEE_no_physical_RLM', 'DL_NEE_RLM',], 'NEE'),
        ('Ground GPP', ['XGB_GPP_RLM','RF_GPP_RLM','DL_GPP_0_RLM_Single', 'DL_GPP_no_physical_RLM', 'DL_GPP_RLM'], 'GPP'),
        ('Ground RECO', ['XGB_RECO_RLM','RF_RECO_RLM','DL_RECO_0_RLM_Single','DL_RECO_no_physical_RLM','DL_RECO_RLM'], 'RECO'),
        ('Ground NEE', ['XGB_NEE','RF_NEE','DL_NEE_0_RL_Single', 'DL_NEE_no_physical', 'DL_NEE'], 'NEE without Meteorological Data'),
        ('Ground GPP', ['XGB_GPP','RF_GPP', 'DL_GPP_0_RL_Single', 'DL_GPP_no_physical','DL_GPP'], 'GPP without Meteorological Data'),
        ('Ground RECO', ['XGB_RECO','RF_RECO', 'DL_RECO_0_RL_Single','DL_RECO_no_physical', 'DL_RECO'], 'RECO without Meteorological Data')
    ]

    for idx, (ax, (ground_col, model_cols, title)) in enumerate(zip(axs.flatten(), groups)):
        plot_model_biases(ax, idx, ground_col, model_cols, title)

    fig.subplots_adjust(right=0.9,bottom=0.2)

    fig.text(0.03, 0.75, '(a) Input with meteorological data', ha='center', va='center', rotation='vertical',
             fontsize=23, fontname='Arial')
    fig.text(0.03, 0.3, '(b) Input without meteorological data', ha='center', va='center', rotation='vertical',
             fontsize=23, fontname='Arial')

    # ✅ 创建统一的图例
    handles = [plt.Line2D([0], [0], linestyle='-', marker='o', color=colors[i], label=model) for i, model in
               enumerate(model_subtitle)]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0), ncol=len(model_subtitle), fontsize=18, columnspacing=1.6, markerscale=2)

    # 调整布局，避免图例被遮挡
    plt.tight_layout(rect=[0, 0.01, 1, 0.95]) #rect=[0, 0.08, 1, 0.95]
    # Adjust overall layout
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.13, top=0.95, wspace=0.2, hspace=0.25)
    # # 调整布局
    results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Bias/'
    # 保存与显示
    plt.savefig(results_path + 'Bias of each model for all models—250614.png', dpi=300, bbox_inches='tight')
    plt.show()
