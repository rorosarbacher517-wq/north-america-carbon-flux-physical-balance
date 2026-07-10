import numpy as np
import os


# 给定影像的根路径
if __name__ == "__main__":
    # download the data
    inputdata_path = "D:/Carbon_flux/data/sites_images_interpolation/"
    x_ref_array = np.load(inputdata_path + 'x_ref_modis_new_mark_interpolation.npy', allow_pickle=True)
    x_lai_array = np.load(inputdata_path + 'x_lai_modis_new_mark_interpolation.npy', allow_pickle=True)
    x_mete_array = np.load(inputdata_path + 'x_mete_era5_mark_interpolation.npy', allow_pickle=True)
    y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
    Y_qa_all = np.load(inputdata_path + 'All_sites_y_qa.npy', allow_pickle=True)
    x_ref_array[:,:,14,1,1]
    y_image_array_all[:,:,0]
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
    # # ************************************
    # 删除meteorological 一年的数据全为-9999的情况
    x_all_invalid_indices22 = []
    for n in range(x_mete_array22.shape[0]):
        if np.all(x_mete_array22[n, ...] == -9999):
            # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
            x_all_invalid_indices22.append(n)
    x_ref_array33 = x_ref_array22[~np.in1d(np.arange(x_ref_array22.shape[0]), x_all_invalid_indices22)]
    x_lai_array33 = x_lai_array22[~np.in1d(np.arange(x_lai_array22.shape[0]), x_all_invalid_indices22)]
    x_mete_array33 = x_mete_array22[~np.in1d(np.arange(x_mete_array22.shape[0]), x_all_invalid_indices22)]
    y_mete_array33 = y_mete_array22[~np.in1d(np.arange(y_mete_array22.shape[0]), x_all_invalid_indices22)]
    print(y_mete_array33.shape)
    print(x_ref_array33.shape)
    print(x_lai_array33.shape)
    x_all_invalid_indices33 = [516]
    x_ref_array44 = x_ref_array33[~np.in1d(np.arange(x_ref_array33.shape[0]), x_all_invalid_indices33)]
    x_lai_array44 = x_lai_array33[~np.in1d(np.arange(x_lai_array33.shape[0]), x_all_invalid_indices33)]
    x_mete_array44 = x_mete_array33[~np.in1d(np.arange(x_mete_array33.shape[0]), x_all_invalid_indices33)]
    y_mete_array44 = y_mete_array33[~np.in1d(np.arange(y_mete_array33.shape[0]), x_all_invalid_indices33)]
    Y_qa_all44 = Y_qa_all[~np.in1d(np.arange(Y_qa_all.shape[0]), x_all_invalid_indices33)]
    x_mete_array44[:,:,0,1,1]
    output_path = 'D:/Carbon_flux/data/sites_images_input/'
    # 检查文件夹是否存在
    if not os.path.isdir(output_path):
        # 创建文件夹
        os.makedirs(output_path)
        print(f"已创建文件夹：{output_path}")
    else:
        print(f"文件夹已存在：{output_path}")

    np.save(output_path + 'x_ref_modis_interpolation_mark.npy', x_ref_array44)
    np.save(output_path + 'x_lai_modis_interpolation_mark.npy', x_lai_array44)
    np.save(output_path + 'x_mete_era5_interpolation_mark.npy', x_mete_array44)
    np.save(output_path + 'y_flux_meteorological_mark.npy', y_mete_array44)
    np.save(output_path + 'All_sites_y_qa.npy', Y_qa_all44)