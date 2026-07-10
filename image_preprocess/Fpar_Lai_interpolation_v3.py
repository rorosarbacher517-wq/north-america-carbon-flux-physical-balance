####本版本是在Fpar_Lai_interpolation版本的基础上，改动了插值的部分：用质量好的像素值去替换质量差的或为-9999的像素

# Fpar_Lai_interpolation
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator

# 通过3*3窗口内的有效值进行线性插值
# def lai_interpolate_window(window, interpolated_b_array):
#     # Convert non-zero values to -9999
#     window[window != 0] = -9999
#     # Convert values of -9999 to NaN
#     window[window == -9999] = np.nan
#
#     # Get the indices of NaN and non-NaN values
#     nan_indices = np.argwhere(np.isnan(window))
#     non_nan_indices = np.argwhere(~np.isnan(window))
#
#     for b in range(interpolated_b_array.shape[0]):
#         # For windows with more than one non-NaN value, use nearest neighbor interpolation
#         if len(non_nan_indices) > 1 and len(non_nan_indices) < 9:
#             x = non_nan_indices[:, 1]
#             y = non_nan_indices[:, 0]
#             values = interpolated_b_array[b, y, x]
#             values_reshaped = values.reshape(-1, 1)
#             non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)
#
#             # Create a nearest neighbor interpolator
#             interp = NearestNDInterpolator(non_nan_indices_reshaped, values_reshaped)
#
#             # Perform nearest neighbor interpolation for NaN values
#             interpolated_values = interp(nan_indices)
#
#             # Assign the interpolated values to the NaN positions in the current window
#             interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values.flatten()
#
#         # For windows with only one non-NaN value, replace NaN values with the non-NaN value
#         elif len(non_nan_indices) == 1:
#             non_nan_value = interpolated_b_array[b, non_nan_indices[0, 0], non_nan_indices[0, 1]]
#             interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = non_nan_value
#         elif len(non_nan_indices) == 9:
#             interpolated_b_array[b, :, :] = interpolated_b_array[b, :, :]
#     return interpolated_b_array

def lai_interpolate_window(window, interpolated_b_array):
    # Convert non-zero values to -9999
    window[window != 0] = -9999
    # Convert values of -9999 to NaN
    window[window == -9999] = np.nan

    # Get the indices of NaN and non-NaN values
    nan_indices = np.argwhere(np.isnan(window))
    non_nan_indices = np.argwhere(~np.isnan(window))

    for b in range(interpolated_b_array.shape[0]):
        # For windows with more than one non-NaN value, use nearest neighbor interpolation
        if len(non_nan_indices) > 1 and len(non_nan_indices) < 9:
            x = non_nan_indices[:, 1]
            y = non_nan_indices[:, 0]
            values = interpolated_b_array[b, y, x]
            values_reshaped = values.reshape(-1, 1)
            non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)

            # Create a nearest neighbor interpolator
            interp = NearestNDInterpolator(non_nan_indices_reshaped, values_reshaped)

            # Perform nearest neighbor interpolation for NaN values
            interpolated_values = interp(nan_indices)

            # Assign the interpolated values to the NaN positions in the current window
            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values.flatten()

        # For windows with only one non-NaN value, replace NaN values with the non-NaN value
        elif len(non_nan_indices) == 1:
            non_nan_value = interpolated_b_array[b, non_nan_indices[0, 0], non_nan_indices[0, 1]]
            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = non_nan_value
        elif len(non_nan_indices) == 9:
            interpolated_b_array[b, :, :] = interpolated_b_array[b, :, :]
    return interpolated_b_array

def check_imagemaxmin(x_iamges_array):
    for i in range(x_iamges_array.shape[2]):
        band_values = x_iamges_array[:, :, i, :, :]  # 选择特定的波长数组
        finite_values_after_replacement = band_values[band_values != -9999]
        max_values_after_replacement = np.nanmax(finite_values_after_replacement)
        min_values_after_replacement = np.nanmin(finite_values_after_replacement)
        print(f"波段 {i + 1} 除去-9999以外的最大值为: {max_values_after_replacement}, 最小值为: {min_values_after_replacement}")

def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

# 给定影像的根路径
if __name__ == "__main__":
    inputdata_path = 'E:/Paper code/code/Carbon_flux_estimation/data/sites_inputdata/preinput_data/'
    fpar_lai_array = np.load(inputdata_path + 'x_lai_array_modis_0_270_mark.npy', allow_pickle=True)
    # check_imagemaxmin(fpar_lai_array)

    print(fpar_lai_array.shape)
    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = fpar_lai_array.copy()
    #####  针对每个波段，每一天的3*3窗口内的值进行插值
    # 应该先把一个时间步长3*3窗口内的低质量值用高质量值进行替换，这样就只有高质量值和-9999了
    # 对每个时间步长进行处理
    for i in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for j in range(interpolated_array.shape[1]):
            # 提取当前质量标识的窗口
            window = interpolated_array[i, j, 2, :, :]
            # 对每个波段，每个窗口内的值进行插值，用窗口内的有效值来插值代替nan值
            # 确认窗口内是否存在nan值并进行插值
            if (0 in window)  and ((1 in window) or (-9999 in window)):
                interpolated_array[i, j, :, :, :] = lai_interpolate_window(window,interpolated_array[i, j, :, :, :])
            else:
                interpolated_array[i, j, :, :, :] = interpolated_array[i, j, :, :, :]
    interpolated_array[0, :, 2, 0, 0]

    check_imagemaxmin(interpolated_array)
    # 插值完之后 ，将mark标记仍为1的视为-9999
    interpolated_array[:, :, 2:, :, :][interpolated_array[:, :, 2:, :, :] == 1] = -9999
    check_imagemaxmin(interpolated_array)
    # # 对每个波段进行每一年的时间序列插值"
    for yy in range(interpolated_array.shape[0]):
        for jj in range(interpolated_array.shape[3]):  # 遍历第二个维度
            for kk in range(interpolated_array.shape[4]):  # 遍历第三个维度
                mark_values = interpolated_array[yy, :, 2, jj, kk]
                # 创建一个索引数组，包含所有非-9999和坏数据值的索引
                mark_valid_indices = np.where(mark_values == 0)[0]
                # 待插值的数据索引,质量标识为1 低质量像素
                interpolate_indices = np.where(interpolated_array[yy, :, 2, jj, kk] != 0)[0]
                for ii in range(interpolated_array.shape[2]):# 遍历第一个维度
                    if len(mark_valid_indices) > 1 and len(mark_valid_indices) < 366:
                        if ii == 0:
                            fillvalue_bounds = (0,1)
                        elif ii == 1:
                            fillvalue_bounds = (0, 10)
                        else:
                            fillvalue_bounds = "extrapolate"
                        # 对于只有一条有效记录的数据，线性插值没有意义，不需要进行插值
                        # 这里需要针对每个波段分别建立插值函数
                        f = interpolate.interp1d(mark_valid_indices, interpolated_array[yy, :, ii, jj, kk][mark_valid_indices],
                                                 kind='linear',bounds_error=False,
                                                 fill_value=fillvalue_bounds)

                        interpolated_values = f(np.arange(len(interpolate_indices )))
                        # 使用索引数组进行插值，填充数据中的空值
                        # interpolated_values = f(np.arange(len(interpolated_array[yy, :, ii, jj, kk])))
                        # 将插值的结果保存到新数组中
                        interpolated_array[yy, :, ii, jj, kk][interpolate_indices] = interpolated_values
                    elif len(mark_valid_indices) == 1:
                        # valid_value = mark_values[mark_valid_indices[0]]
                        interpolated_array[yy, :, ii, jj, kk][interpolate_indices] = interpolated_array[yy, :, ii, jj, kk][mark_valid_indices[0]]
                        # 如果只有一条有效记录的数据，将所有的数值设为该有效记录的值
                    else:
                        # Set all the days of the year to -9999
                        interpolated_array[yy, :, ii, jj, kk] = interpolated_array[yy, :, ii, jj, kk]
    # interpolated_array[:, :, 2, 0, 1]
    check_imagemaxmin(interpolated_array)
    # 保存插值后的数据
    interpolated_array[:, :, 1:2, :, :][interpolated_array[:, :, 1:2, :, :] < 0] = -9999
    # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
    for iii in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for jjj in range(interpolated_array.shape[1]):
            for mmm in range(interpolated_array.shape[3]):
                # 对每个3*3窗口进行处理
                for nnn in range(interpolated_array.shape[4]):
                    # 如果interpolated_array[iii, jjj, :, mmm, nnn]里有任何一个值为-9999，则把interpolated_array[iii, jjj, :, mmm, nnn]所有值都设置为-9999
                    if np.any(interpolated_array[iii, jjj, :, mmm, nnn]== -9999.0) :
                        interpolated_array[iii, jjj, :, mmm, nnn] = -9999
                    else:
                        continue
    check_dim_fill(interpolated_array)
    check_imagemaxmin(interpolated_array)
    np.save(inputdata_path + 'x_lai_array_modis_0_270_mark_interpolation_v3.npy', interpolated_array)

    fpar_lai_array[:, :, 2,1, 1]