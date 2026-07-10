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
import Model_build_attention
import myloss_attention
import other_process
import importlib

importlib.reload(myloss_attention)
importlib.reload(other_process)

base_name = "test"
if '__file__' in globals():
    # base_name = os.path.basename(__file__)+socket.gethostname()
    base_name = os.path.basename(__file__)[15:]


def add_gaussian_noise_to_features(data, noise_std=0.1, feature_start=0, feature_end=9):
    """
    对数据的指定特征维度添加高斯噪声，并自动避开-9999的无效值
    Args:
        data (np.ndarray): 输入数组，形状应为 (N, T, C, H, W)
        noise_std (float): 噪声标准差，控制噪声强度
        feature_start (int): 起始特征索引（包含）
        feature_end (int): 终止特征索引（不包含）
    Returns:
        np.ndarray: 添加噪声后的数组，形状与输入一致
    """
    # 深拷贝原始数据防止污染
    noisy_data = data.copy()

    # 提取目标特征切片（第2维的 [feature_start:feature_end] 特征）
    target_slice = noisy_data[:, :, feature_start:feature_end, :, :]

    # 创建有效值掩码（True表示有效位置）
    valid_mask = (target_slice != -9999)

    # 生成高斯噪声（仅在有效位置添加）
    gaussian_noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=target_slice.shape
    ) * valid_mask  # 关键步骤：屏蔽无效位置

    # 叠加噪声（自动广播到完整维度）
    target_slice += gaussian_noise

    # 回写修改后的特征数据
    noisy_data[:, :, feature_start:feature_end, :, :] = target_slice

    return noisy_data

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

# 给定影像的根路径
if __name__ == "__main__":
    EPOCH = 100
    # EPOCH = 70
    # LEARNING_RATE = 0.001 # 0.001(OK),0.0001,0.01,0.1,
    LEARNING_RATE = 0.01
    BATCH_SIZE = 16  #
    L2 = 0.0001  # 0.0001(ok),0.00001,0.001,0.01

    # EPOCH           =   int(sys.argv[1] )
    # LEARNING_RATE   = float(sys.argv[2])
    # BATCH_SIZE      =   int(sys.argv[3])
    # L2              = float(sys.argv[4])

    # download the data
    base_name = base_name + ".epoch{:03d}.rate{:.5f}.batch{:03d}.L{:.5f}".format(EPOCH, LEARNING_RATE, BATCH_SIZE, L2)
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
    x_inter_mask_all[..., 0, 1, 1]
    x_inter_mask_all
    print('hhh')
    ##### 匹配前检查同一像素不同波段值缺失的情况
    # inputdata_path = './data/sites_results/'
    # x_image_array_all = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    # y_image_array_all = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    # # y_image_array_all = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    # x_inter_mask_all = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    # y_qa_array_all = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)

    # 将y_image_array_all进行特征填充
    # x_image_array is 7 reflectance + 2 lai/fpar + 6 ERA5 ()
    x_image_array, y_image_array, x_inter_mask_array0, y_qa_array = Data_filter_match.mul_match_x_y_modis_quality(
        x_image_array_all,
        y_image_array_all,
        x_inter_mask_all,
        y_qa_array_all,
        ref=True, lai=True,
        meteorology=False,  # flux site meteorology
        era5_meteorology=True,  # era5 meteorology
        NEE_GPP_RECO=True,
        ref_quality=0,
        ref_filledvalue=0,
        lai_filledvalue=0,
        lai_quality=0,
        era5_mete_filledvalue=0,
        meteorology_filledvalue=0,
        NEE_GPP_RECO_filledvalue=0,is_inter=0)
    print(x_image_array.shape, y_image_array.shape)
    x_inter_mask_array = other_process.full_intervalue(x_inter_mask_array0)
    y_flux_qa_array = y_qa_array[:, :, -5:]
    x_image_array[:, :, 0, 0, 1]
    y_image_array[:,:, 7]
    x_inter_mask_array[203, :, :, 0, 1]
    y_flux_qa_array[:,:,3]
    # 将交叉验证的结果保存下来进行验证分析
    # results_variation_path = './data/sites_sensitivity/'
    results_variation_path = './data/sites_modis_noise/'
    if not os.path.isdir(results_variation_path):
        # 创建文件夹
        os.makedirs(results_variation_path)
        print(f"已创建文件夹：{results_variation_path}")
    np.save(results_variation_path + 'DL_RLM_x_mask_attention.npy', x_inter_mask_array)
    np.save(results_variation_path + 'DL_RLM_y_qa_attention.npy',y_flux_qa_array)  # y_flux_qa_array.shape (1688, 365, 5), the fouth dimension is QA
    # 将质量标识的波段去除
    # x_image_array = x_image_array[:,:,[0,1,2,3,4,5,6,14,16,17,18,19,20,21,22],:,:]
    covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12, 3, 5, 16, 17, 18, 19, 20, 21, 22]]
    other_process.check_dim_fill(x_image_array)
    other_process.y_statistic_df(y_image_array)
    # 得到三组数据 分别用
    # 将covariate_array中的植被类型（新array中的第8列）和气候类型（新array中的第8列）换成离散数字表示，因为字符串类型计算不了均值等统计值
    covariate_array = other_process.replace_veg_cli(covariate_array)

    # 根据modis 几个波段的数据添加高斯噪声
    # new_array = calculate_vegetation_indices(x_image_array)
    # new_array[:,:,15,1,1]
    ### 向数据集中添加噪声
    # np.random.seed(42)
    # input_images_train_noisy = add_gaussian_noise_to_features(
    #     x_image_array,
    #     noise_std=0.01,
    #     feature_start=0,
    #     feature_end=9
    # )
    # # assert np.array_equal(input_images_train_norm0, input_images_train_noisy)  # 应成立
    # # 检查训练集噪声范围
    # noise_diff = x_image_array[:, :, :9] - input_images_train_noisy[:, :, :9]
    # valid_mask = (x_image_array[:, :, :9] != -9999)
    # print("实际噪声标准差:", np.std(noise_diff[valid_mask]))  # 应接近设定的noise_std
    # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
    # input_images_train_norm0 = other_process.process_pixel_values(input_images_train_norm0)
    # input_images_test_norm0 = other_process.process_pixel_values(input_images_test_norm0)
    # 仅选取MODIS 太阳辐射 和 空气温度
    # input_images_train_noisy[:, :, 0, 1, 1]
    # x_image_array[:, :, 0, 1, 1]
    # # k倍交叉验证划分数据集
    cross_validation_data = Data_preprocess.get_cross_validation_training_test(x_image_array, y_image_array,
                                                                               covariate_array)
    # print(cross_validation_data) # # 统计输入模型的数据-9999所占的比例
    # loss = mask_loss(maskValue=-9999)
    importlib.reload(myloss_attention)
    # loss = myloss.mask_loss(maskValue=-9999) # for multiple/single variable without physics
    loss = myloss_attention.mask_loss_physical(maskValue=-9999) # for multiple variable with physics
    # loss = myloss.mask_loss_huber(maskValue=-9999)  # for multiple variable with physics
    # loss = myloss.mask_loss_physical_MAE(maskValue=-9999) # for multiple variable with physics
    METHOD = 2  # 2(OK),0,1

    # IS_TEST = 1
    # ITERS = 5
    print('Data is ready!')
    Y_pred_filtered = []
    Y_test_filtered = []
    Y_test_sites = []
    Y_attention = []
    Rmse = []
    R2 = []
    CC = []
    Bias = []
    N = []
    # stacked_array = None
    # import Model_build_copy
    import Model_build_attention

    # importlib.reload(Model_build_copy)
    importlib.reload(Model_build_attention)
    import Model_history_attention

    importlib.reload(Model_history_attention)
    # print(input_images_test_norm3)
    # layer_cnn=3,2(0k);layern=2,3;units=32(ok),64,16;n_head=6,5(ok),4,3;drop=0.001(ok),0.0001,0.00001,0.01,0.1;doy_encoder=2,0(ok),1
    # doy_encoder=0;
    doy_encoder = 2;
    # drop = 0.1; units=64
    drop = 0.1;
    # units = 32（ok）
    units = 64  # _3 RLFM
    # units = 32 # _4 没有气象变量 # 单独预测的GPP GPP GPP
    # y_SCALE = 0.1
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
    importlib.reload(Model_build_attention)

    model = Model_build_attention.get_transformer_cnn3(layer_cnn=1, layern=2, units=units,
                                                       n_times=input_images_pre_norm3.shape[1],
                                                       n_window=3, n_head=4, drop=drop, n_out=3, is_batch=True,
                                                       mask_value=-9999.0, is_rptv=True, is_sensor=False)
    print(model.summary())  # 输出模型各层的参数状况
    input_images_train_norm3 = np.array(input_images_pre_norm3)
    input_covariate_train_norm3 = np.array(input_covariate_train_norm0)
    model_history = Model_history_attention.my_train_1schedule_time_drop(model, input_images_pre_norm3, prey_norm3,
        epochs=EPOCH, start_rate=LEARNING_RATE, loss=loss,
        per_epoch=per_epoch, split_epoch=5, option=METHOD, decay=L2,
        batch_size=BATCH_SIZE, validation_split=0.04, hold_epoch=0,
        reduce_epoch=False, gaps_p=0.1)

    model_pre_name = f'./region model/get_cnn_transformer_RLM_0420_all.h5'
    model_pre2 = Model_build_attention.get_transformer_cnn3(layer_cnn=1, layern=2, units=units,
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
