# 随机森林同时预测5倍交叉验证
import gc
import os
import tensorflow as tf
import region_prediction_preprocess
from rasterio.crs import CRS
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import shutil

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
def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

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

def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))


def fill_nearest_neighbor_values(array, target_value, filter_size=3):
    filled_array = array.copy()
    filled_array[:, :, 0]
    # Define the mask for the target values
    replace_mask = (filled_array == target_value)
    # Get the average value of the non-target neighbors using a mean filter
    neighbor_average = uniform_filter(filled_array, size=filter_size, mode='constant')
    # Fill the target values with the neighbor averages
    filled_array[replace_mask] = neighbor_average[replace_mask]
    return filled_array


def print_predictions_for_window_condition(predictions, dataset, condition_value=-9999):
    # Get the shape of the dataset
    samples, timesteps, rows, cols, bands = dataset.shape
    # Iterate through the data to check the condition in each 3x3 window
    for i in range(samples):
        window_values = dataset[i, :, :, :, :]
        if np.all(window_values == -9999):
            print(f"Predicted values for window at position ({i}):")
            print(predictions[i, :])


# 通过3*3窗口内的有效值进行线性插值
def lai_interpolate_window(window, interpolated_b_array):
    # Convert non-zero values to -9999
    window[window == 1.000e-08] = -9999
    # Convert window array to float type
    window = window.astype(float)
    # Convert values of -9999 to NaN
    window[window == -9999] = np.nan

    # Get the indices of NaN and non-NaN values
    nan_indices = np.argwhere(np.isnan(window))
    non_nan_indices = np.argwhere(~np.isnan(window))

    # For windows with more than one non-NaN value, use nearest neighbor interpolation
    if len(non_nan_indices) > 1 and len(non_nan_indices) < 1024*366:
        x = non_nan_indices[:, 1]
        y = non_nan_indices[:, 0]
        values = interpolated_b_array[y, x, b]
        values_reshaped = values.reshape(-1, 1)
        non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)

        # Create a nearest neighbor interpolator
        interp = NearestNDInterpolator(non_nan_indices_reshaped, values_reshaped)

        # Perform nearest neighbor interpolation for NaN values
        interpolated_values = interp(nan_indices)

        # Assign the interpolated values to the NaN positions in the current window
        interpolated_b_array[nan_indices[:, 0], nan_indices[:, 1], b] = interpolated_values.flatten()

    # For windows with only one non-NaN value, replace NaN values with the non-NaN value
    elif len(non_nan_indices) == 1:
        non_nan_value = interpolated_b_array[non_nan_indices[0, 0], non_nan_indices[0, 1], b]
        interpolated_b_array[nan_indices[:, 0], nan_indices[:, 1], b] = non_nan_value
    elif len(non_nan_indices) == 1024*366:
        # Do nothing for the entire window
        pass

    return interpolated_b_array

# def replace_zero_values(array):
#     interpolated_array = np.copy(array)  # Create an interpolated array same as the original array
#     for b in range(7,9):
#         for i in range(array.shape[3]):
#             for j in range(array.shape[4]):
#                 inter_array = array[:, :, b, i, j]
#                 if np.any(inter_array == 1.000e-08):
#                     # Using the lai_interpolate_window function for nearest neighbor interpolation
#                     interpolated_values = lai_interpolate_window(inter_array, interpolated_array)
#                     interpolated_array[:, :] = interpolated_values
#     return interpolated_array



# 给定影像的根路径
if __name__ == "__main__":
    # 加载模型
    model_pre = tf.keras.models.load_model('./region model/get_cnn_transformer_RLM_cross.h5', compile=False)
    # download the data
    # 所有region 存储的根目录 每个国家的文件夹
    # input_path
    #input_path = 'D:/'
    region_base_dir = 'E:/The North America ecosystem carbon flux/Region_data/'

    out_base_dir = os.path.join(region_base_dir, 'output_to_geotif')
    if not os.path.isdir(out_base_dir):

        os.makedirs(out_base_dir)
    # 进入国家级下面的州级目录

    nation_path = os.path.join(region_base_dir, 'blocks_to_input_npy')

    # 每个州每个block存储的根路径
    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    state_name = state_list[1]
    #state_name = 'Montana'
    state_path = os.path.join(nation_path, state_name)

    output_state_path = os.path.join(out_base_dir, state_name)
    # bands_mean_std_path = os.path.join(bands_mean_std_dir,state_name+'.csv')
    bands_mean_std_df = pd.read_csv("D:/Code/Carbon_flux/region model/sites_statistic/Image_eigenvalues_pre.csv")
    # Extract the columns ending with '_mean' and '_stdDev'
    # mean_cols = [col for col in bands_mean_std_df.columns if col.endswith('_mean')]
    # stdDev_cols = [col for col in bands_mean_std_df.columns if col.endswith('_stdDev')]
    # Extract the values and convert them into arrays
    mean_array = np.array(bands_mean_std_df['mean_train'].values).reshape(-1, 1, 1)
    stdDev_array = np.array(bands_mean_std_df['std_train'].values).reshape(-1, 1, 1)

    if not os.path.isdir(output_state_path):
        os.makedirs(output_state_path)

    # 进入州下面的block目录
    blocklist = os.listdir(state_path)
    # blocklist = ['block_0_0','block_0_1','block_0_2', 'block_0_3', 'block_0_4', 'block_0_5','block_0_6',
    #              'block_1_0','block_1_1','block_1_2', 'block_1_3', 'block_1_4', 'block_1_5','block_1_6',
    #              'block_2_0','block_2_1','block_2_2', 'block_2_3', 'block_2_4', 'block_2_5','block_2_6',
    #              'block_3_0', 'block_3_1', 'block_3_2', 'block_3_3', 'block_3_4', 'block_3_5', 'block_3_6',
    #              'block_4_0', 'block_4_1', 'block_4_2', 'block_4_3', 'block_4_4', 'block_4_5', 'block_4_6',
    #              'block_5_0', 'block_5_1', 'block_5_2', 'block_5_3', 'block_5_4', 'block_5_5', 'block_5_6'
    #              ]
    # blocklist = ['block_0_6','block_0_7','block_1_6','block_1_7']
    for b in range(0,len(blocklist)): # len(blocklist)
        block = blocklist[b]
        # output_block_path = os.path.join(output_state_path, block)
        # if not os.path.isdir(output_block_path):
        #     os.makedirs(output_block_path)
        # 读取各个变量
        blocks_path = os.path.join(state_path, block)
        x_ref_array_path = blocks_path + '/' + block + '_ref_input3.npy'
        x_lai_array_path = blocks_path + '/' + block + '_lai_input3.npy'
        x_mete_array_path = blocks_path + '/' + block + '_mete_input3.npy'
        y_image_array_all_path = blocks_path + '/' + block + '_spatialtime_input3.npy'
        # 打开之前保存的地理信息文件
        geoinfo_file_path = os.path.join(blocks_path, f'{block}_geoinfo.txt')
        # 检查四个文件是否存在，如果不存在则跳过处理下一个文件
        if not all(os.path.exists(path) for path in
                   [x_ref_array_path, x_lai_array_path, x_mete_array_path, y_image_array_all_path,geoinfo_file_path]):
            continue  # 如果其中任何一个文件不存在，则跳过处理下一个文件
        x_ref_array = np.load(x_ref_array_path, allow_pickle=True)
        x_lai_array = np.load(x_lai_array_path, allow_pickle=True)
        x_mete_array = np.load(x_mete_array_path, allow_pickle=True)
        y_image_array_all = np.load(y_image_array_all_path, allow_pickle=True)
        #
        x_image_array_all = np.concatenate((x_ref_array[:,:,:,:,:], x_lai_array[:,:,:,:,:], x_mete_array[:,:,:,:,:]), axis=2)
        # 选取x_image_array_all[]中ref lai mete的波段
        x_image_array_select = x_image_array_all[:, :, [0, 1, 2, 3, 4, 5, 6,14,15,17,18,19,20,21,22], :, :] #14,15,
        # 统一为各个波段为-9999的情况
        x_ref_array[:, :,1, 1, 1]
        x_image_array_re = full_filledvalue(x_image_array_select)
        x_image_array_re[:, :, 6, 1, 1]
        # check_dim_fill(x_image_array_re)
        x_image_array = process_pixel_values(x_image_array_re)
        # x_image_array = region_prediction_preprocess.mul_match_x_y_modis(x_image_array_all, ref=True,
        #                     lai=False, era5_meteorology=True, ref_quality=0, ref_filledvalue=0,
        #                     lai_filledvalue=0, lai_quality=0, era5_mete_filledvalue=0)
        covariate_array = y_image_array_all[:,:,:]
        # 对x_image_array 进行标准化处理
        # input_images_pre_norm0, input_covariate_pre_norm0, mean_pre, std_pre, mean_pre_cov, std_pre_cov = region_prediction_preprocess.pre_construct_composite_norm(
        #     x_image_array, covariate_array, is_single_norm=True)
        input_images_pre_norm0, input_covariate_pre_norm0, mean_pre, std_pre, mean_pre_cov, std_pre_cov = region_prediction_preprocess.pre_construct_composite_norm(
            x_image_array, covariate_array,mean_array,stdDev_array,is_single_norm=True)
        # 对x_image_array 进行标准化处理
        print(input_images_pre_norm0.shape)
        # 将x_image_array进行标准化
        # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
        # input_images_pre_norm0 = process_pixel_values(input_images_pre_norm0)
        # input_covariate_pre_norm0 = process_pixel_values(input_covariate_pre_norm0)
        # 复制数据 变换维度
        input_images_pre_norm3 = np.transpose(input_images_pre_norm0, (0, 1, 3, 4, 2))
        input_covariate_pre_norm3 = input_covariate_pre_norm0
        # input_images_pre_norm3[:, :, 6, 1, 1]
        input_images_pre_norm3 = tf.convert_to_tensor(input_images_pre_norm3, dtype=tf.float32)
        input_covariate_pre_norm3 = tf.convert_to_tensor(input_covariate_pre_norm3, dtype=tf.float32)
        # 进行预测
        pre_datasetx = input_images_pre_norm3[:, :365, :, :, :]
        y_pred_filtered = model_pre.predict(pre_datasetx)
        # 将这个数组按天计算求均值 即可得到这个像素在这一年内所有天的均值
        # 沿着第一维（索引为1的维度）计算均值
        mean_y_pred = np.mean(y_pred_filtered, axis=1)
        # print(mean_y_pred)
        # 打印当pre_datasetx 所有波段均为-9999时，对应的三个变量的预测值
        # Define the function to the predictions for the specified condition in each 3x3 window
        # Call the function with the predictions and pre_datasetx
        print_predictions_for_window_condition(mean_y_pred, input_images_pre_norm3)
        ###*******************************
        # 将预测的结果与相应的地理数据结合起来 输出成tif
        # 将预测的数组reshape为32，32，3的格式，与其相应的crs和transform对应起来输出为geotif格式
        mean_y_pred
        mean_y_pred_reshaped = mean_y_pred.reshape(32, 32, 3)
        mean_y_pred_reshaped[:,:,0]
        # # 选择每个像素对应的中心经纬度坐标
        mean_y_pred_geolat_lon = y_image_array_all[:, 0, 11:13].reshape(32, 32, 2)

        # 打开之前保存的地理信息文件
        # geoinfo_file_path = os.path.join(blocks_path, f'{block}_geoinfo.txt')
        with open(geoinfo_file_path, 'r') as file:
            lines = file.readlines()

        # 提取仿射变换参数和投影信息
        output_origin = eval(lines[0].split(': ')[1])
        output_pixel_size = eval(lines[1].split(': ')[1])

        # 根据之前保存的地理信息创建仿射变换对象
        transform = Affine.from_gdal(output_origin[0], output_pixel_size[0], 0.0, output_origin[1], 0.0,
                                     -output_pixel_size[1])

        # 设置CRS
        crs = CRS.from_epsg(4326)

        # 保存重整形后的数组为单波段GeoTIFF并命名
        output_files = ['NEE_RLM.tif', 'GPP_RLM.tif', 'RECO_RLM.tif']

        for i, output_file in enumerate(output_files):
            with rasterio.open(os.path.join(output_state_path, f'{block}_{output_file}'), 'w', driver='GTiff', height=mean_y_pred_reshaped.shape[0],
                               width=mean_y_pred_reshaped.shape[1], count=1, dtype=mean_y_pred_reshaped.dtype, crs=crs,
                               transform=transform) as dst:
                dst.write(mean_y_pred_reshaped[:, :, i], indexes=1)

        print('The ' +block + '单波段GeoTIFF文件已成功创建并命名')
        gc.collect()
        # 使用shutil.rmtree来删除blocks_path以及其中的所有文件和子目录
        shutil.rmtree(blocks_path)

    # inputdata_path = 'E:/Carbon_flux/data/regoin_image/region_images_test_blocks_input/Aguascalientes/block_0_0/'
    # x_ref_array = np.load(inputdata_path + 'block_0_0_ref_input.npy', allow_pickle=True)
    # x_lai_array = np.load(inputdata_path + 'block_0_0_lai_input.npy', allow_pickle=True)
    # x_mete_array = np.load(inputdata_path + 'block_0_0_mete_input.npy', allow_pickle=True)
    # y_image_array_all = np.load(inputdata_path + 'block_0_0_spatialtime_input.npy', allow_pickle=True)
    #
    # print('kkkk')
    # geoinformation = 'E:/Carbon_flux/data/regoin_image/region_images_test_blocks/Aguascalientes/geotransformation/block_0_0.txt'
    # x_image_array_all = np.concatenate((x_ref_array, x_lai_array, x_mete_array), axis=2)
    # # 选取x_image_array_all[]中ref lai mete的波段
    # x_image_array = x_image_array_all[:,:,[0,1,2,3,4,5,6,14,15,17, 18, 19, 20, 21, 22],:,:]
    # # 统一为各个波段为-9999的情况
    # x_image_array = full_filledvalue(x_image_array)
    # covariate_array = y_image_array_all
    # # 对x_image_array 进行标准化处理
    # input_images_pre_norm0,input_covariate_pre_norm0,mean_pre,std_pre,mean_pre_cov,std_pre_cov = region_prediction_preprocess.pre_construct_composite_norm(x_image_array,covariate_array,is_single_norm = True)
    # # 对x_image_array 进行标准化处理
    # print(input_images_pre_norm0.shape)
    # # 将x_image_array进行标准化
    # # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
    # input_images_pre_norm0 = process_pixel_values(input_images_pre_norm0)
    # # input_covariate_pre_norm0 = process_pixel_values(input_covariate_pre_norm0)
    # # 复制数据 变换维度
    # input_images_pre_norm3 = np.transpose(input_images_pre_norm0, (0, 1, 3, 4, 2))
    # input_covariate_pre_norm3 = input_covariate_pre_norm0
    #
    # input_images_pre_norm3 = tf.convert_to_tensor(input_images_pre_norm3, dtype=tf.float32)
    # input_covariate_pre_norm3 = tf.convert_to_tensor(input_covariate_pre_norm3, dtype=tf.float32)
    # pre_datasetx = [input_images_pre_norm3, input_covariate_pre_norm3]
    # # 进行预测
    # model_pre = tf.keras.models.load_model('./model_data/get_transformer_cnn_total.h5', compile=False)
    # y_pred_filtered = model_pre.predict(pre_datasetx)
    # # 将这个数组按天计算求均值 即可得到这个像素在这一年内所有天的均值
    # # 沿着第一维（索引为1的维度）计算均值
    # mean_y_pred = np.mean(y_pred_filtered, axis=1)
    # print(mean_y_pred)
    # # 将预测的数组reshape为32，32，3的格式，与其相应的crs和transform对应起来输出为geotif格式
    # import numpy as np
    # import rasterio
    # from rasterio.transform import Affine
    #
    # # 重整形mean_y_pred数组为（32，32，3）
    # mean_y_pred_reshaped = mean_y_pred.reshape(32, 32, 3)
    # # 选择每个像素对应的中心经纬度坐标
    # mean_y_pred_geolat_lon = y_image_array_all[:,0,11:13].reshape(32, 32, 2)
    # # Define the transformation and CRS
    # # Load the geo information file
    # # with open('E:/Carbon_flux/data/regoin_image/region_images_test_blocks/Aguascalientes/geotransformation/block_0_0.txt', 'r') as file:
    # #     content = file.readlines()
    # #
    # #     transform = from_origin(mean_y_pred_geolat_lon.min(), mean_y_pred_geolat_lon.max(), 1.0,
    # #                             1.0)  # Replace these values with actual coordinates
    # #     crs_info = int(content[3].split()[-1])
    # #     crs = CRS.from_epsg(crs_info)
    # # Define the transformation and CRS
    # # 定义Affine转换参数
    # with open('E:/Carbon_flux/data/regoin_image/region_images_test_blocks/Aguascalientes/geotransformation/block_0_0.txt', 'r') as file:
    #     content = file.readlines()
    # transform = Affine.from_gdal(-102.87506633736761, 0.004491576420597608, 0.0, 22.462373679408635, 0.0,
    #                              -0.004491576420597608)
    #
    # # 设置CRS
    # crs = CRS.from_epsg(4326)
    #
    # # 保存重整形后的数组为单波段GeoTIFF并命名
    # output_files = ['output_nee.tif', 'output_gpp.tif', 'output_reco.tif']
    #
    # for i, output_file in enumerate(output_files):
    #     with rasterio.open(f'./data/{output_file}', 'w', driver='GTiff', height=mean_y_pred_reshaped.shape[0],
    #                        width=mean_y_pred_reshaped.shape[1], count=1, dtype=mean_y_pred_reshaped.dtype, crs=crs,
    #                        transform=transform) as dst:
    #         dst.write(mean_y_pred_reshaped[:, :, i], indexes=1)
    #
    # print("单波段GeoTIFF文件已成功创建并命名。")
    #
    # # # Save the reshaped array as a GeoTIFF
    # # with rasterio.open('output.tif', 'w', driver='GTiff', height=mean_y_pred_reshaped.shape[0],
    # #                    width=mean_y_pred_reshaped.shape[1], count=mean_y_pred_reshaped.shape[2],
    # #                    dtype=mean_y_pred_reshaped.dtype, crs=crs, transform=transform) as dst:
    # #     for i in range(mean_y_pred_reshaped.shape[2]):
    # #         dst.write(mean_y_pred_reshaped[:, :, i], indexes=i + 1)
    #
    # print("GeoTIFF file with georeferencing and coordinates has been successfully created.")
    #
    # print("GeoTIFF 文件已成功创建。")