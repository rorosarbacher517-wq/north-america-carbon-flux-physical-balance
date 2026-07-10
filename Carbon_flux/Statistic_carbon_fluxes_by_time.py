# 统计每个时间段缺失的数据比例
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

    inputdata_path = './data/sites_input/'

    x_ref_array = np.load(inputdata_path + 'x_ref_modis_interpolation_mark.npy',
                          allow_pickle=True)  # the dimension 15 indicate whether to interpolate
    # 15 bands: 0-6 reflectance; 7-13 QA; 14 is interpolation indication (0 is measrued and 1 is interpolated)
    x_lai_array = np.load(inputdata_path + 'x_lai_modis_interpolation_mark.npy',
                          allow_pickle=True)  # 4 bands: 2 bands for LAI, fpar; 1 band QA; 1 band interpolation indication
    x_mete_array = np.load(inputdata_path + 'x_mete_era5_interpolation_mark.npy',
                           allow_pickle=True)  # ERA 5 meteorological 7 bands; 0-5 meteorological; 6  interpolation indication
    y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
    y_qa_array_all = np.load(inputdata_path + 'All_sites_y_qa.npy', allow_pickle=True)
    x_image_array_all = np.concatenate(
        (x_ref_array[..., :-1, :, :], x_lai_array[..., :-1, :, :], x_mete_array[..., :-1, :, :]),
        axis=2)  # 23 bands all but not including interpolation indication
    x_inter_mask_all = np.concatenate(
        (x_ref_array[..., -1:, :, :], x_lai_array[..., -1:, :, :], x_mete_array[..., -1:, :, :]),
        axis=2)  # 3 interpolation indication

#**********************************************
    # 统计每天缺失的数据量
    # 创建变量字典，访问需要的数据
    def create_variable_definition():
        return {
            'Doy_data': y_image_array_all[:, :, 4].flatten(),  # 扁平化 DOY 数据
            'NEE_data': y_image_array_all[:, :, 6].flatten(),  # 扁平化 NEE 数据
            'MODIS_data': x_inter_mask_all[..., 0, 1, 1].flatten()  # 扁平化 NEE 数据
        }


    variables = create_variable_definition()


    # 缺失数据统计函数
    def calculate_nan_statistics(day_data,nee_data):
        # total_count = nee_data.size
        total_count = day_data.size  # 该月的数据量
        missing_count = np.sum(day_data == -9999)
        missing_ratio = missing_count / total_count if total_count > 0 else 0  # 避免除以零
        return missing_count, missing_ratio


    # 统计每天的缺失数据并保存为 CSV
    def get_daily_statistics_and_save(nee_data, doy_data, output_file):
        results = []

        # total_records = doy_data.shape[0]  # 记录总数（1688）

        for doy in range(1, 366):  # 从 1 到 365 遍历 DOY
            # 提取属于当前 DOY 的 NEE 数据
            day_mask = doy_data == doy  # 布尔索引，获取当前 DOY 的索引
            day_nee = nee_data[day_mask]  # 提取对应的 NEE 数据

            # 统计缺失数据
            nee_missing_count, nee_missing_ratio = calculate_nan_statistics(day_nee,nee_data)

            # 存储结果
            results.append({
                'DOY': doy,
                'Missing Count': nee_missing_count,
                'Missing Ratio': nee_missing_ratio,
            })

        # 创建 DataFrame 并保存为 CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Daily missing data statistics saved to {output_file}")


    # 获取每天的缺失数据统计并保存为 CSV
    NEE_data = variables['NEE_data']
    Doy_data = variables['Doy_data']

    output_csv = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/' + 'daily_missing_data_statistics 0407.csv'
    get_daily_statistics_and_save(NEE_data, Doy_data, output_csv)
#**********************************************
    # # 统计每月缺失的数据量
    # # 创建变量字典，访问需要的数据
    # def create_variable_definition():
    #     return {
    #         'Month_data': y_image_array_all[:, :, 5].flatten(),  # 扁平化年数据
    #         'NEE_data': y_image_array_all[:, :, 6].flatten(),  # 扁平化 NEE 数据
    #         'MODIS_data': x_inter_mask_all[..., 0, 1, 1].flatten()  # 扁平化 MODIS 数据
    #     }
    #
    #
    # # 创建变量字典
    # variables = create_variable_definition()
    #
    #
    # # 缺失数据统计函数
    # def calculate_nan_statistics(data, modis_data):
    #     # total_count = modis_data.size
    #     total_count = data.size  # 该月的数据量
    #     missing_count = np.sum(data == -9999)  # 假设缺失数据用 np.nan 表示
    #     missing_ratio = missing_count / total_count if total_count > 0 else 0  # 避免除以零
    #     return missing_count, missing_ratio
    #
    #
    # # 统计每年的缺失数据并保存为 CSV
    # def get_month_statistics_and_save(modis_data, month_data, output_file):
    #     results = []
    #
    #     unique_months = np.unique(month_data)  # 获取唯一年份
    #     for month in unique_months:
    #         month_mask = month_data == month  # 布尔索引，获取当前年份的索引
    #         month_modis_data = modis_data[month_mask]  # 提取对应的 MODIS 数据
    #
    #         # 统计缺失数据
    #         modis_missing_count, modis_missing_ratio = calculate_nan_statistics(month_modis_data, modis_data)
    #
    #         # 存储结果
    #         results.append({
    #             'Month': month,
    #             'Missing Count': modis_missing_count,
    #             'Missing Ratio': modis_missing_ratio,
    #         })
    #
    #     # 创建 DataFrame 并保存为 CSV
    #     df = pd.DataFrame(results)
    #     df.to_csv(output_file, index=False)
    #     print(f"Annual missing data statistics saved to {output_file}")
    #
    #
    # # 获取每年的缺失数据统计并保存为 CSV
    # MODIS_data = variables['NEE_data']
    # Month_data = variables['Month_data']
    #
    # output_csv = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/' + 'Monthly_missing_data_statistics 0407.csv'
    # get_month_statistics_and_save(MODIS_data, Month_data, output_csv)
#**********************************************
    # # 统计每年缺失的数据量
    # # 创建变量字典，访问需要的数据
    # def create_variable_definition():
    #     return {
    #         'Year_data': y_image_array_all[:, :, 3].flatten(),  # 扁平化年数据
    #         'NEE_data': y_image_array_all[:, :, 6].flatten(),  # 扁平化 NEE 数据
    #         'MODIS_data': x_inter_mask_all[..., 0, 1, 1].flatten()  # 扁平化 MODIS 数据
    #     }
    #
    #
    # # 创建变量字典
    # variables = create_variable_definition()
    #
    #
    # # 缺失数据统计函数
    # def calculate_nan_statistics(data,modis_data):
    #     # total_count = modis_data.size
    #     total_count = data.size
    #     missing_count = np.sum(data == 1)  # 假设缺失数据用 np.nan 表示
    #     missing_ratio = missing_count / total_count if total_count > 0 else 0  # 避免除以零
    #     return missing_count, missing_ratio
    #
    #
    # # 统计每年的缺失数据并保存为 CSV
    # def get_annual_statistics_and_save(modis_data, year_data, output_file):
    #     results = []
    #
    #     unique_years = np.unique(year_data)  # 获取唯一年份
    #     for year in unique_years:
    #         year_mask = year_data == year  # 布尔索引，获取当前年份的索引
    #         year_modis_data = modis_data[year_mask]  # 提取对应的 MODIS 数据
    #
    #         # 统计缺失数据
    #         modis_missing_count, modis_missing_ratio = calculate_nan_statistics(year_modis_data,modis_data)
    #
    #         # 存储结果
    #         results.append({
    #             'Year': year,
    #             'Missing Count': modis_missing_count,
    #             'Missing Ratio': modis_missing_ratio,
    #         })
    #
    #     # 创建 DataFrame 并保存为 CSV
    #     df = pd.DataFrame(results)
    #     df.to_csv(output_file, index=False)
    #     print(f"Annual missing data statistics saved to {output_file}")
    #
    #
    # # 获取每年的缺失数据统计并保存为 CSV
    # MODIS_data = variables['MODIS_data']
    # Year_data = variables['Year_data']
    #
    # output_csv = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/' + 'annual_MODIS_missing_data_statistics 0407.csv'
    # get_annual_statistics_and_save(MODIS_data, Year_data, output_csv)