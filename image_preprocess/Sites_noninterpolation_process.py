import numpy as np
import os

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

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
    # download the data
    inputdata_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_mark/'
    x_ref_array = np.load(inputdata_path + 'x_ref_array_modis_0_271_mark.npy', allow_pickle=True)
    x_lai_array = np.load(inputdata_path + 'x_lai_array_modis_0_271_mark.npy', allow_pickle=True)
    x_mete_array = np.load(inputdata_path + 'x_mete_array_era5_0_271_mark.npy', allow_pickle=True)
    y_image_array_all = np.load(inputdata_path + 'y_flux_array_0_271_mark.npy', allow_pickle=True)

    x_ref_array1 = x_ref_array.copy()
    # 将质量标记为1和-9999的都转换为-9999
    x_ref_array1[:, :, 7:, :, :][x_ref_array1[:, :, 7:, :, :] == 1] = -9999
    # x_ref_array = full_filledvalue(x_ref_array)
    x_ref_array1 = full_filledvalue(x_ref_array1)
    # x_ref_array[:,:,7,1,1]
    # x_ref_array1[:, :, 7, 1, 1]
    x_lai_array[:, :, 2:, :, :][x_lai_array[:, :, 2:, :, :] == 1] = -9999
    x_lai_array = full_filledvalue(x_lai_array)

    x_image_array_all = np.concatenate((x_ref_array, x_lai_array, x_mete_array), axis=2)
    unique_site_ids = set(y_image_array_all[:, 0, 0])
    # **********************************************************************************************************************
    # # 删除特定站点的数据 南美洲 夏威夷6个站点
    delete_stations = ['AR-TF1', 'AR-TF2', 'BR-CST', 'BR-Npw', 'US-SuM', 'US-SuS', 'US-SuW', 'US-xPU']

    y_all_valid_indices22 = []
    for n in range(y_image_array_all.shape[0]):
        if y_image_array_all[n, 0, 0] not in delete_stations:
            y_all_valid_indices22.append(n)
    x_ref_array22 = x_ref_array[y_all_valid_indices22]
    y_mete_array22 = y_image_array_all[y_all_valid_indices22]
    x_lai_array22 = x_lai_array[y_all_valid_indices22]
    x_mete_array22 = x_mete_array[y_all_valid_indices22]
    x_ref_array22[:,:,0,1,1]
    x_lai_array22[:, :, 0, 1, 1]
    x_mete_array22[:, :, 0, 1, 1]
    print(y_mete_array22.shape)
    print(x_ref_array22.shape)
    print(x_lai_array22.shape)

    output_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_input/non_interpolation/'
    # 检查文件夹是否存在
    if not os.path.isdir(output_path):
        # 创建文件夹
        os.makedirs(output_path)
        print(f"已创建文件夹：{output_path}")
    else:
        print(f"文件夹已存在：{output_path}")

    np.save(output_path + 'x_ref_array_modis_0_263_noninterpolation.npy', x_ref_array22)
    np.save(output_path + 'x_lai_array_modis_0_263_noninterpolation.npy', x_lai_array22)
    np.save(output_path + 'x_mete_array_era5_0_263_noninterpolation.npy', x_mete_array22)
    np.save(output_path + 'y_flux_array_0_263_nonmark.npy', y_mete_array22)


