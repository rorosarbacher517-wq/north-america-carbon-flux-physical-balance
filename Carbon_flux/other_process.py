# other_process
import os 
import numpy as np 
import pandas as pd

def y_statistic_df(y_image_array):
    # 提取除为字符串的第0 第16,17列以外的所有列
    data = y_image_array[:, :, np.r_[1:16, 18:23]]
    data = data.astype(float)  # 将数据类型转换为浮点数
    data = data.reshape(-1, 20)  # 将数据展平为二维数组，每行20个特征
    # 忽略等于-9999的填充值
    data[data == -9999.0] = np.nan
    # 计算均值、标准差、最大值和最小值（忽略NaN值）
    mean_values = np.nanmean(data, axis=0)
    std_values = np.nanstd(data, axis=0)
    max_values = np.nanmax(data, axis=0)
    min_values = np.nanmin(data, axis=0)
    # 创建 DataFrame 对象
    df_data = pd.DataFrame({
        'mean': mean_values,
        'std': std_values,
        'max': max_values,
        'min': min_values
    })
    # 输出 DataFrame
    statistic_dir = './data/sites_statistic/'
    if not os.path.isdir(statistic_dir):
        # 创建文件夹
        os.makedirs(statistic_dir)
        print(f"已创建文件夹：{statistic_dir}")
    else:
        print(f"文件夹已存在：{statistic_dir}")
    df_data.to_csv(statistic_dir + 'y_statistic.csv')
    # print(df_data)


def replace_veg_cli(covariate_array):
    # covariate_array[covariate_array == -9999] = -9999.0
    unique_values_veg = np.unique(covariate_array[:, :, 7][covariate_array[:, :, 7] != -9999].astype(str))
    unique_values_clim = np.unique(covariate_array[:, :, 8][covariate_array[:, :, 8] != -9999].astype(str))
    replacement_dict_veg = {value: index for index, value in enumerate(unique_values_veg)}
    replacement_dict_clim = {value: index for index, value in enumerate(unique_values_clim)}
    # 将这个字典输出保存为txt
    statistic_dir = './data/sites_statistic/'
    if not os.path.isdir(statistic_dir):
        # 创建文件夹
        os.makedirs(statistic_dir)
        print(f"已创建文件夹：{statistic_dir}")
    else:
        print(f"文件夹已存在：{statistic_dir}")
    with open(os.path.join(statistic_dir, "replacement_dict_veg.txt"), "w") as f:
        for key, value in replacement_dict_veg.items():
            f.write(f"{key}:{value}\n")
    with open(os.path.join(statistic_dir, "replacement_dict_clim.txt"), "w") as f:
        for key, value in replacement_dict_clim.items():
            f.write(f"{key}:{value}\n")
    for i in range(covariate_array.shape[0]):
        for j in range(covariate_array.shape[1]):
            if covariate_array[i, j, 7] != -9999.0:
                covariate_array[i, j, 7] = replacement_dict_veg[str(covariate_array[i, j,7])]
            if covariate_array[i, j, 8] != -9999.0:
                covariate_array[i, j, 8] = replacement_dict_clim[str(covariate_array[i, j, 8])]
    return covariate_array


def process_pixel_values(input_array):
    # output_array = np.zeros_like(input_array)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            # for m in range(input_array.shape[2]):
            window_values = input_array[i, j, :, :, :]
            if np.any(window_values != -9999):
                input_array[i, j, :, :, :] = np.where(window_values == -9999, 0, window_values)
            else:
                input_array[i, j, :, :, :] = -9999
    return input_array


## real,pred = y_test_plot_i, y_pred_plot_i
def get_regression_line(real, pred, data_range=(0, 110)):
    # 拟合（若换MK，自行操作）最小二乘
    def slope(xs, ys):
        m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
        b = np.mean(ys) - m * np.mean(xs)
        return m, b

    k, b = slope(real, pred)
    regression_linex = []
    regression_liney = []
    for a in range(int(data_range[0]), int(data_range[1]) + 1):
        # print (a)
        regression_liney.append(a)
        regression_linex.append((k * a) + b)
    return regression_linex, regression_liney


def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999.0).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

def get_inter_array(x_ref_array,is_inter):
    # 创建一个与x_ref_array形状相同的mask数组，并初始化为-9999
    nointer_array = np.full_like(x_ref_array[...], -9999)
    inter_array = np.full_like(x_ref_array[...], -9999)
    all_array = x_ref_array.copy()
    # 根据最后一个特征值创建mask
    for i in range(x_ref_array.shape[0]):
        for j in range(x_ref_array.shape[1]):
            for m in range(x_ref_array.shape[3]):
                for n in range(x_ref_array.shape[4]):
                    if x_ref_array[i, j, -1, m, n] == 0:
                        nointer_array[i, j, :, m, n] = np.where(nointer_array[i, j, :, m, n] == -9999,
                                                                x_ref_array[i, j, :, m, n], -9999)
                    elif x_ref_array[i, j, -1, m, n] == 1:
                        inter_array[i, j, :, m, n] = np.where(inter_array[i, j, :, m, n] == -9999,
                                                              x_ref_array[i, j, :, m, n], -9999)
                    # 对于特征值为2的情况，保持不变
                    else:
                        all_array[i, j, :, m, n] = x_ref_array[i, j, :, m, n]
    if is_inter == 0:
        x_array = nointer_array[..., :-1, :, :]
    elif is_inter == 1:
        x_array = inter_array[..., :-1, :, :]
    else:
        x_array = all_array[..., :-1, :, :]
    return x_array


def full_intervalue(interpolated_array):
    invalid_indices = np.any(
        (interpolated_array[:, :, :, :, :] == -9999) | (interpolated_array[:, :, :, :, :] == 1), axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

def full_y_fillvalue(interpolated_array):
    # Check for invalid indices
    invalid_indices = np.any((interpolated_array[:, :, 4:5] == -9999) | (interpolated_array[:, :, 6:7] == -9999)
                             | (interpolated_array[:, :, 7:8] == -9999)| (interpolated_array[:, :, 8:9] == -9999), axis=2)

    # Create reshaped indices
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)

    # Set all 23 bands to -9999 where the condition is True
    interpolated_array[reshaped_indices] = -9999

    return interpolated_array

def interpolation_quality(DL_Y_predict,DL_Y_true,inter_mask,DL_y_qa,is_inter = 0, threshold=1):
    # x:插值与不插值；y:good quality 和other
    DL_mask = DL_Y_true[:, :, 0] == -9999
    inter_mask_condition = inter_mask[:, :, 0, 1, 1] == -9999
    DL_y_qa_mask = (DL_y_qa[:, :, 3] < 0) | (DL_y_qa[:, :, 3] > threshold)
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
    return DL_NEE,DL_GPP,DL_RECO,Ground_NEE,Ground_GPP,Ground_RECO,combined_mask


