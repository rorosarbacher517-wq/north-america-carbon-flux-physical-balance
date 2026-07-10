# 随机森林同时预测5倍交叉验证

import importlib
import pandas as pd
import os
import Data_preprocess
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
from keras import backend as K
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import Data_filter_match


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
    statistic_dir = './region model/sites_statistic/'
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
    statistic_dir = './region model/sites_statistic/'
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

def full_filledvalue(interpolated_array):
    # 插值完之后 ，将mark标记仍为1的视为-9999  这种情况是由于整个年时间序列366天当中没有一个像素是质量好的，所以仍会出现质量为1的情况
    # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
    for iii in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for jjj in range(interpolated_array.shape[1]):
            for mmm in range(interpolated_array.shape[3]):
                # 对每个3*3窗口进行处理
                for nnn in range(interpolated_array.shape[4]):
                    if np.any(interpolated_array[iii, jjj, :, mmm, nnn] == -9999):
                        interpolated_array[iii, jjj, :, mmm, nnn] = -9999
                    else:
                        continue
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

def mask_loss(maskValue=-9999.0):
    def mask_loss_in(y_true, y_pred):
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        y_pred = y_pred * isMask
        axis_to_reduce = range(1, K.ndim(y_true))
        sum_square = K.sum(K.square(y_true - y_pred), axis=axis_to_reduce)
        sum_n = K.sum(isMask, axis=axis_to_reduce)
        loss1 = K.sqrt(sum_square / sum_n)
        return loss1

    return mask_loss_in


# maskValue=-9999
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# y_pred = model.predict(inputs)
# y_true = tf.convert_to_tensor(trainy[:2,:,:].astype(np.float))
# y_true = trainy[:2,:,:].astype(np.float)
def mask_loss_physical(maskValue=-9999.0):
    lambda1 = 0.1
    lambda1 = 1

    # lambda1 = 10
    def mask_loss_in(y_true, y_pred):
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        # loss #1
        y_pred1 = y_pred * isMask
        axis_to_reduce = range(1, K.ndim(y_true))
        sum_square1 = K.sum(K.square(y_true - y_pred1), axis=axis_to_reduce)
        sum_n1 = K.sum(isMask, axis=axis_to_reduce)
        loss1 = K.sqrt(sum_square1 / sum_n1)
        # loss #2
        isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
        sum_square2 = K.sum(K.square(y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]) * isMask2, axis=1)
        sum_n2 = K.sum(isMask2, axis=1)
        loss2 = K.sqrt(sum_square2 / sum_n2)
        return loss1 + lambda1 * loss2

    return mask_loss_in


# maskValue=-9999
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# y_pred = model.predict(inputs)
# y_true = tf.convert_to_tensor(trainy[:2,:,:].astype(np.float32))
# y_true = trainy[:2,:,:].astype(np.float32)
def mask_loss_physical_MAE(maskValue=-9999.0):
    lambda1 = 0.1
    lambda1 = 1

    # lambda1 = 10
    def mask_loss_in(y_true, y_pred):
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        # loss #1
        y_pred1 = y_pred * isMask
        axis_to_reduce = range(1, K.ndim(y_true))
        sum_square1 = K.sum(K.abs(y_true - y_pred1), axis=axis_to_reduce)
        sum_n1 = K.sum(isMask, axis=axis_to_reduce)
        loss1 = sum_square1 / sum_n1
        # loss #2
        isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
        sum_square2 = K.sum(K.abs(y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]) * isMask2, axis=1)
        sum_n2 = K.sum(isMask2, axis=1)
        loss2 = sum_square2 / sum_n2
        return loss1 + lambda1 * loss2

    return mask_loss_in

def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999.0).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array


def full_intervalue(interpolated_array):
    invalid_indices = np.any(
        (interpolated_array[:, :, :, :, :] == -9999) | (interpolated_array[:, :, :, :, :] == 1), axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

# 给定影像的根路径
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
        (x_ref_array[..., :-1, :, :], x_lai_array[..., :-1, :, :], x_mete_array[..., :-1, :, :]), axis=2)
    x_inter_mask_all = np.concatenate(
        (x_ref_array[..., -1:, :, :], x_lai_array[..., -1:, :, :], x_mete_array[..., -1:, :, :]), axis=2)
    ##### 匹配前检查同一像素不同波段值缺失的情况
    x_mete_array[:, :, 0, 0, 0]
    y_image_array_all[:, :, 0]
    x_lai_array[:, :, 0, 1, 1]
    y_qa_array_all[:, :, 3]
    # 将y_image_array_all进行特征填充
    x_image_array, y_image_array, x_inter_mask_array0, y_qa_array = Data_filter_match.mul_match_x_y_modis(
        x_image_array_all,
        y_image_array_all,
        x_inter_mask_all,
        y_qa_array_all,
        ref=True, lai=True,
        meteorology=False,
        era5_meteorology=True,
        NEE_GPP_RECO=True,
        ref_quality=0,
        ref_filledvalue=0,
        lai_filledvalue=0,
        lai_quality=0,
        era5_mete_filledvalue=0,
        meteorology_filledvalue=0,
        NEE_GPP_RECO_filledvalue=0)
    print(x_image_array.shape, y_image_array.shape)
    # x_image_array = full_filledvalue(x_image_array)
    x_image_array[:, :, 7, 1, 1]
    y_image_array[:, :, 3]
    x_inter_mask_array = full_intervalue(x_inter_mask_array0)
    y_flux_qa_array = y_qa_array[:, :, -5:]
    y_flux_qa_array[:, :, 0]
    x_inter_mask_array0[:, :, 0, 1, 1]
    # 将交叉验证的结果保存下来进行验证分析
    results_variation_path = './region model'

    if not os.path.isdir(results_variation_path):
        # 创建文件夹
        os.makedirs(results_variation_path)
        print(f"已创建文件夹：{results_variation_path}")
    np.save(results_variation_path + 'DL_RLM_x_mask_region.npy', x_inter_mask_array)
    np.save(results_variation_path + 'DL_RLM_y_qa_region.npy', y_flux_qa_array)
    covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12, 3, 5, 16, 17, 18, 19, 20, 21, 22]]
    # # 统计y_image_array的特征
    y_statistic_df(y_image_array)
    # # 将covariate_array中的植被类型（新array中的第8列）和气候类型（新array中的第8列）换成离散数字表示，因为字符串类型计算不了均值等统计值
    covariate_array = replace_veg_cli(covariate_array)
    # print(cross_validation_data) # # 统计输入模型的数据-9999所占的比例
    EPOCH = 100
    # EPOCH = 70
    # LEARNING_RATE = 0.001 # 0.001(OK),0.0001,0.01,0.1,
    LEARNING_RATE = 0.01
    # loss = mask_loss(maskValue=-9999)
    loss = mask_loss_physical(maskValue=-9999)
    # loss = mask_loss_physical_MAE(maskValue=-9999)
    BATCH_SIZE = 16  # 16(OK),32,64,128,8
    METHOD = 2  # 2(OK),0,1
    L2 = 0.0001  # 0.0001(ok),0.00001,0.001,0.01
    IS_TEST = 1
    ITERS = 5
    print('Data is ready!')
    Y_pred_filtered = []
    Y_test_filtered = []
    Y_test_sites = []
    Rmse = []
    R2 = []
    CC = []
    Bias = []
    N = []
    stacked_array = None
    # import Model_build_copy
    import Model_build
    # importlib.reload(Model_build_copy)
    importlib.reload(Model_build)
    import Model_history
    importlib.reload(Model_history)
    doy_encoder = 2;
    drop = 0.1;
    units = 64
    y_SCALE = 1
    # 根据index划分训练集和测试集
    index_train, index_test, trainx, trainy, testx, testy, train_covariate,test_covariate,train_sites_counts,test_sites_counts,RF_trainy,RF_testy = Data_preprocess.get_training_test_com2\
        (x_image_array,y_image_array,covariate_array,proportion=0.8,split_by_site=True)
    trainy = trainy[:, :, :].reshape(trainy.shape[0], trainy.shape[1], 3)
    testy = testy[:, :, :].reshape(testy.shape[0], testy.shape[1], 3)
    # ## 对训练集和测试集的影像数据和其他协变量进行标准化
    input_images_train_norm0, input_images_test_norm0, input_covariate_train_norm0, input_covariate_test_norm0, mean_train, std_train, mean_train_cov, std_train_cov = Data_preprocess.construct_composite_train_test(
        trainx,
        testx, train_covariate, test_covariate, is_train_test_com=True, is_single_norm=True)
    # 将训练集和测试集拼接起来构成一个整个站点数据集的训练集 以此来预测区域级的变量
    input_images_pre_norm1 = np.concatenate((input_images_train_norm0,input_images_test_norm0))
    input_covariate_pre_norm1 = np.concatenate((input_covariate_train_norm0, input_covariate_test_norm0))
    prey_norm3 =np.concatenate((trainy, testy))
    # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
    input_images_pre_norm2 = process_pixel_values(input_images_pre_norm1)
    # 复制数据 变换维度
    input_images_pre_norm3 = np.transpose(input_images_pre_norm2, (0, 1, 3, 4, 2))
    input_images_pre_norm3 = np.array(input_images_pre_norm3)
    # input_covariate_pre_norm3 = np.array(input_covariate_pre_norm1)
    train_n = input_images_pre_norm3.shape[0]
    per_epoch = train_n // BATCH_SIZE
    start = time.time()
    importlib.reload(Model_build)
    model = Model_build.get_transformer_cnn2(layer_cnn=1, layern=2, units=units,
                                             n_times=input_images_pre_norm3.shape[1],
                                             n_window=3, n_head=4, drop=drop, n_out=3, is_batch=True,
                                             mask_value=-9999.0, is_rptv=True, is_sensor=False)
    print(model.summary())  # 输出模型各层的参数状况
    input_images_train_norm3 = np.array(input_images_pre_norm3)
    input_covariate_train_norm3 = np.array(input_covariate_train_norm0)
    model_history = Model_history.my_train_1schedule_time_drop(model, input_images_pre_norm3, prey_norm3, epochs=EPOCH,
                                                               start_rate=LEARNING_RATE, loss=loss, per_epoch=per_epoch,
                                                               split_epoch=5,
                                                               option=METHOD, decay=L2, batch_size=BATCH_SIZE,
                                                               validation_split=0.04, hold_epoch=0, reduce_epoch=False,
                                                               gaps_p=0.1)

    model_pre_name = './region model/get_cnn_transformer_RLM_0304.h5'
    model_pre2 = Model_build.get_transformer_cnn2(layer_cnn=1, layern=2, units=units,
                                             n_times=input_images_pre_norm3.shape[1],
                                             n_window=3, n_head=4, drop=drop, n_out=3, is_batch=True,
                                             mask_value=-9999.0, is_rptv=True, is_sensor=False)

    layers = model_pre2.layers if len(model_pre2.layers) < len(model.layers) else model.layers
    for il, ilayer in enumerate(layers):
        ilayer1 = model_pre2.layers[il]
        ilayer2 = model.layers[il]
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic != '_'])
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic != '_'])
        if name_cls == name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            print("\t" + ilayer.name, end=" ")
            model_pre2.layers[il].set_weights(model.layers[il].get_weights())

    model_pre2.save(model_pre_name)
