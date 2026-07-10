import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator
import pandas as pd

def ref_interpolate_window(interpolated_b_array):
    for b in range(interpolated_b_array.shape[0]):
        if b >= 0 and b <= 6:
            bb = b + 7
        else:
            bb = b
        window = interpolated_b_array[bb, :, :]
        window[window != 0] = -9999  # Convert non-zero values to -9999
        window[window == -9999] = np.nan  # Convert -9999 to NaN

        nan_indices = np.argwhere(np.isnan(window))  # Get the indices of NaN values
        non_nan_indices = np.argwhere(~np.isnan(window))  # Get the indices of non-NaN values

        if len(non_nan_indices) > 1 and len(non_nan_indices) < 9:
            x = non_nan_indices[:, 1]
            y = non_nan_indices[:, 0]
            values = interpolated_b_array[b, y, x].reshape(-1, 1)
            non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)

            interp = NearestNDInterpolator(non_nan_indices_reshaped, values)  # Create nearest neighbor interpolator
            interpolated_values = interp(nan_indices)  # Perform nearest neighbor interpolation for NaN values

            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:,
                                                       1]] = interpolated_values.flatten()  # Assign the interpolated values to the NaN positions
        elif len(non_nan_indices) == 1:
            non_nan_value = interpolated_b_array[b, non_nan_indices[0, 0], non_nan_indices[0, 1]]
            interpolated_b_array[
                b, nan_indices[:, 0], nan_indices[:, 1]] = non_nan_value  # Replace NaN values with the non-NaN value
        elif len(non_nan_indices) == 9:
            interpolated_b_array[b, :, :] = interpolated_b_array[b, :, :]
    return interpolated_b_array


def ref_time_series_interpolation(array_data, band_index,fillvalue_bounds):
    if band_index in [0,1,2,3,4,5,6]:
        array_data = array_data.astype(float)
        array_data[array_data == -9999] = np.nan
        df_band = pd.DataFrame(array_data)
        df_band_linear = df_band.interpolate(method='linear', axis=1, limit_direction='both')
        # 处理0-1之外的除了-9999之外的值为空值然后使用nearest插值
        # Step 1: Select rows based on the condition
        condition = (df_band_linear < fillvalue_bounds[0]) | (df_band_linear > fillvalue_bounds[1])
        # Step 2: Replace the selected values with -9999
        df_band_linear[condition] = np.nan
        df_band_linear = df_band_linear.fillna(-9999)
        array_inter = np.array(df_band_linear)
    else:
        array_data = array_data
        # 将array_data中的-9999值用0替换
        array_data[array_data == -9999] = 0
        array_inter = array_data
    return array_inter

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array
# def full_filledvalue(interpolated_array):
#     # 插值完之后 ，将mark标记仍为1的视为-9999  这种情况是由于整个年时间序列366天当中没有一个像素是质量好的，所以仍会出现质量为1的情况
#     # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
#     for iii in range(interpolated_array.shape[0]):
#         # 对每个3*3窗口进行处理
#         for jjj in range(interpolated_array.shape[1]):
#             for mmm in range(interpolated_array.shape[3]):
#                 # 对每个3*3窗口进行处理
#                 for nnn in range(interpolated_array.shape[4]):
#                     if np.any(interpolated_array[iii, jjj, :, mmm, nnn] == -9999):
#                         interpolated_array[iii, jjj, :, mmm, nnn] = -9999
#                     else:
#                         continue
#     return interpolated_array

def check_imagemaxmin(x_iamges_array):
    for i in range(x_iamges_array.shape[2]):
        band_values = x_iamges_array[:, :, i, :, :]  # 选择特定的波长数组
        finite_values_after_replacement = band_values[band_values != -9999]
        max_values_after_replacement = np.nanmax(finite_values_after_replacement)
        min_values_after_replacement = np.nanmin(finite_values_after_replacement)
        print(f"波段 {i + 1} 除去-9999以外的最大值为: {max_values_after_replacement}, 最小值为: {min_values_after_replacement}")

def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999.0).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

# 给定影像的根路径
if __name__ == "__main__":
    inputdata_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_mark/'
    ref_array = np.load(inputdata_path + 'x_ref_array_modis_0_271_mark.npy', allow_pickle=True)
    # check_imagemaxmin(ref_array )

    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = ref_array.copy()
    #####  针对每个波段，每一天的3*3窗口内的值进行插值
    # 应该先把一个时间步长3*3窗口内的低质量值用高质量值进行替换，这样就只有高质量值和-9999了
    # 对每个时间步长进行处理
    # for i in range(interpolated_array.shape[0]):
    #     # 对每个3*3窗口进行处理
    #     for j in range(interpolated_array.shape[1]):
    #         interpolated_array[i, j, :, :, :] = ref_interpolate_window(interpolated_array[i, j, :, :, :])
    # check_imagemaxmin(interpolated_array)
    interpolated_array[:, :, 7:, :, :][interpolated_array[:, :, 7:, :, :] == 1] = -9999
    interpolated_array = full_filledvalue(interpolated_array)
    interpolated_array2 = interpolated_array.copy()
    # 对每个波段进行每一年的时间序列插值"
    for band_index in range(interpolated_array2.shape[2]):
        for ii in range(interpolated_array2.shape[3]):
            for jj in range(interpolated_array2.shape[4]):
                array_data = interpolated_array2[:, :, band_index, ii, jj]
                if band_index < 7:
                    fillvalue_bounds = (0, 1)
                else:
                    fillvalue_bounds = "extrapolate"
                interpolated_array2[:, :, band_index, ii, jj] = ref_time_series_interpolation(array_data,
                                                                                              band_index,
                                                                                              fillvalue_bounds)
    # 将空值用-9999来填充
    interpolated_array2[np.isnan(interpolated_array2)] = -9999
    interpolated_array2 = full_filledvalue(interpolated_array2)
    check_dim_fill(interpolated_array2)
    check_imagemaxmin(interpolated_array2)
    output_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_interpolation/'
    # 保存插值后的数据
    np.save(output_path + 'x_ref_array_modis_0_271_interpolation.npy', interpolated_array2)