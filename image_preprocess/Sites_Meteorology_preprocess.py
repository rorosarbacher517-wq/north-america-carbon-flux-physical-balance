import numpy as np
import pandas as pd

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

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

def mete_time_series_interpolation(array_data, band_index,fillvalue_bounds):
    array_data[array_data==0] = -9999
    array_data = array_data.astype(float)
    array_data[array_data == -9999] = np.nan
    df_band = pd.DataFrame(array_data)
    df_band_linear = df_band.interpolate(method='linear', axis=1, limit_direction='both')
    df_band_linear = df_band_linear.fillna(-9999)
    array_inter = np.array(df_band_linear)
    return array_inter


# 给定影像的根路径
if __name__ == "__main__":
    inputdata_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_mark/'
    era5_array = np.load(inputdata_path + 'x_mete_array_era5_0_271_mark.npy', allow_pickle=True)
    check_imagemaxmin(era5_array)

    interpolated_array = era5_array.copy()

    interpolated_array[:, :, 2:, :, :][interpolated_array[:, :, 2:, :, :] == 1] = -9999
    interpolated_array = full_filledvalue(interpolated_array)
    interpolated_array2 = interpolated_array.copy()
    for band_index in range(interpolated_array2.shape[2]):
        for ii in range(interpolated_array2.shape[3]):
            for jj in range(interpolated_array2.shape[4]):
                array_data = interpolated_array2[:, :, band_index, ii, jj]
                fillvalue_bounds = "extrapolate"
                interpolated_array2[:, :, band_index, ii, jj] = mete_time_series_interpolation(array_data,
                                                                                              band_index,
                                                                                              fillvalue_bounds)
    # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
    for iii in range(interpolated_array2.shape[0]):
        # 对每个3*3窗口进行处理
        for jjj in range(interpolated_array2.shape[1]):
            for mmm in range(interpolated_array2.shape[3]):
                # 对每个3*3窗口进行处理
                for nnn in range(interpolated_array2.shape[4]):
                    if np.any(interpolated_array2[iii, jjj, :, mmm, nnn] == -9999):
                        interpolated_array2[iii, jjj, :, mmm, nnn] = -9999
                    else:
                        continue
    check_dim_fill(interpolated_array2)
    check_imagemaxmin(interpolated_array2)
    output_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_interpolation/'
    # 保存插值后的数据
    np.save(output_path + 'x_mete_array_era5_0_271_interpolation.npy', interpolated_array2)