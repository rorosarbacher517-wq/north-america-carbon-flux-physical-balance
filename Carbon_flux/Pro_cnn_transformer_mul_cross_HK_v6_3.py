# 检查土壤水含量对模型建模的影响
# v6_3 same as !v6_2! but single estimating each flux
# v5_2 same as !v5_1! but using 64
# !v5_1! same as v!4_8! but using drop in training - drop help little
# v5_0 1-layer T-block for meterlogical and 2-layer T blocks + indepedent decoder blocks;
# v4_7 2-layer T blocks;  v!4_8! 1-layer T-block for meterlogical and 2-layer T blocks;  v4_9 3-layer T-blocks
# v4_5 Hank revised based on v4_4 copy send by Bin
# v4_4 Hank revised based on v4_3 copy send by Bin on Apr 18, 2024

# cnn transformer 同时预测5倍交叉验证
# v1_2 Hank fix out1 = attn_output in Transformer block
# v1_3 fix a small issue in loss function and drop=0.1
# v1_4 use drop after CNN and use 64 - slightly decrease
# v1_5 use drop after CNN and use 32 - slightly decrease so botj 64 and drop after CNN not help
# v1_6 same as v1_3 but use validation_split = 0.04 (fix a bug)
# v1_7/v1_8 same as v1_6 but try doy_encoder = 1 and 2 (2 is good)
# v1_9 change doy 2 model to be combined input of doy and rpv and use two layer with one relu for doy and rpv
# v2_0 change doy 2 model to be combined input of doy and rpv and use two layer with two relu for doy and rpv
# little diff with v1_9 but slightly worse
# v2_1 same as v2_0 but add scale parameters - a bug appears
# v2_2 !!! same as v2_0 but add scale parameters and scale =10 & good reduced bais
# v2_3 same as v2_2 but use physical with 0.1
# v2_4 same as v2_2 but use physical with 1.0
# v2_5 same as v2_2 but use physical with 10
# v2_6 !!! same as v2_2 (all linear units) - use physical with 1.0
# v2_7 same as v2_4 but use linear, exp and exp loss
# v2_8 same as v2_4 but use exp, exp and exp loss
# v2_9/3_0 same 2_8 use 0.01 and 0.0001 rate; 0.01 rate works fine but exp for nee does not work
# v3_1 !!! same as v2_7 linear    , exp and exp loss use 0.01 rate
# v3_2/3/4 same as v2_7 linear*exp, exp and exp loss use 0.01/0.001.0.05 rate - not good
# v3_5 same as v2_7 linear    , exp and exp loss use 0.05 rate
# v3_6 same as v2_7 linear    , exp and exp loss use 0.01 rate and 70e
# v3_7 !!! same as v3_1
# v3_8 same as v3_1 but physical with 0.1 does not work
# v3_9 !!! does not use scale - works OK
# v4_0 not scale and pysical 10 does not work
# v4_1 !!! same as v3_9 looks better
# v4_2 !!! same as v4_1 but using MAE
# v4_3 !!! same as v4_2 but using the meteorology gridded image

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

import myloss
import other_process
import importlib

importlib.reload(myloss)
importlib.reload(other_process)

base_name = "test"
if '__file__' in globals():
    # base_name = os.path.basename(__file__)+socket.gethostname()
    base_name = os.path.basename(__file__)[15:]

# import Data_filter_match_v1

import sys

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
        (x_ref_array[..., :-1, :, :], x_lai_array[..., :-1, :, :], x_mete_array[..., :-1, :, :]),axis=2)  # 23 bands all but not including interpolation indication
    x_inter_mask_all = np.concatenate(
        (x_ref_array[..., -1:, :, :], x_lai_array[..., -1:, :, :], x_mete_array[..., -1:, :, :]),axis=2)  # 3 interpolation indication
    ##### 匹配前检查同一像素不同波段值缺失的情况
    # 将y_image_array_all进行特征填充
    # x_image_array is 7 reflectance + 2 lai/fpar + 6 ERA5 ()
    x_image_array, y_image_array, x_inter_mask_array0, y_qa_array = Data_filter_match.mul_match_x_y_modis(
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
        NEE_GPP_RECO_filledvalue=0)
    print(x_image_array.shape, y_image_array.shape)
    x_inter_mask_array = other_process.full_intervalue(x_inter_mask_array0)
    y_flux_qa_array = y_qa_array[:, :, -5:]
    x_image_array[:, :, 0, 0, 1]
    y_image_array[:,:, 7]
    x_inter_mask_array[203, :, :, 0, 1]
    # 将交叉验证的结果保存下来进行验证分析
    results_variation_path = './data/sites_results/'
    if not os.path.isdir(results_variation_path):
        # 创建文件夹
        os.makedirs(results_variation_path)
        print(f"已创建文件夹：{results_variation_path}")
    # np.save(results_variation_path + 'DL_RLM_x_mask_MAE_1.npy', x_inter_mask_array)
    # np.save(results_variation_path + 'DL_RLM_y_qa_MAE_1.npy',y_flux_qa_array)
    # y_flux_qa_array.shape (1688, 365, 5), the fouth dimension is QA
    # 将质量标识的波段去除
    # x_image_array = x_image_array[:,:,[0,1,2,3,4,5,6,14,16,17,18,19,20,21,22],:,:]
    other_process.check_dim_fill(x_image_array)
    # ******************************************
    # check_dim_fill(x_image_array)
    # 对不符合条件的噪声值进行预处理
    # covariate_array = y_image_array[:, :, [3, 8, 9, 10, 11, 1, 2, 4, 15, 16, 17, 18, 19, 20, 21]]
    covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12, 3, 5, 16, 17, 18, 19, 20, 21, 22]]

    # # 分别对应DoY，Rg_daily，PotRad_uStar_daily，Tair_f_daily，VPD_f_daily，Year，Month，
    # # Vegetation_Abbreviation(IGBP)，Climate_Class_Abbreviation(Koeppen)，Mean_Average_Precipitation(mm)，
    # # Mean_Average_Temperature(degreesC)，Latitude(degrees)，Longitude(degrees)，Elevation(m)
    # # 统计y_image_array的特征
    other_process.y_statistic_df(y_image_array)
    # ****-->HK Q3: 植被类型和气候类型 - 没有用到吧？
    # # 将covariate_array中的植被类型（新array中的第8列）和气候类型（新array中的第8列）换成离散数字表示，因为字符串类型计算不了均值等统计值
    covariate_array = other_process.replace_veg_cli(covariate_array)
    # # # 根据index划分训练集和测试集
    # index_train, index_test, trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, RF_trainy, RF_testy = Data_preprocess.get_training_test_com2 \
    #     (x_image_array, y_image_array, covariate_array, proportion=0.8, split_by_site=True)
    # print(trainx.shape, trainy.shape, testx.shape, testy.shape)
    # 保存x_image_array, y_image_array, covariate_array 数组
    np.save(results_variation_path + 'DL_RLM_x_image_array.npy', x_image_array)
    np.save(results_variation_path + 'DL_RLM_y_image_array.npy', y_image_array)
    np.save(results_variation_path + 'DL_RLM_covariate_array.npy', covariate_array)
    # # k倍交叉验证划分数据集
    cross_validation_data = Data_preprocess.get_cross_validation_training_test(x_image_array, y_image_array,covariate_array)
    # print(cross_validation_data) # # 统计输入模型的数据-9999所占的比例
    # loss = mask_loss(maskValue=-9999)
    importlib.reload(myloss)
    # loss = myloss.mask_loss(maskValue=-9999) # for multiple/single variable without physics
    # loss = myloss.mask_loss_physical(maskValue=-9999) # for multiple variable with physics
    # loss = myloss.mask_loss_huber(maskValue=-9999) # for multiple variable with physics
    loss = myloss.mask_loss_physical_MAE(maskValue=-9999) # for multiple variable with physics
    METHOD = 2  # 2(OK),0,1

    # IS_TEST = 1
    # ITERS = 5
    print('Data is ready!')
    Y_pred_filtered = []
    Y_test_filtered = []
    Y_test_sites = []
    Rmse = []
    R2 = []
    CC = []
    Bias = []
    N = []
    # stacked_array = None
    # import Model_build_copy
    import Model_build

    # importlib.reload(Model_build_copy)
    importlib.reload(Model_build)
    import Model_history

    importlib.reload(Model_history)
    # print(input_images_test_norm3)
    # layer_cnn=3,2(0k);layern=2,3;units=32(ok),64,16;n_head=6,5(ok),4,3;drop=0.001(ok),0.0001,0.00001,0.01,0.1;doy_encoder=2,0(ok),1
    # doy_encoder=0;
    doy_encoder = 2;
    # drop = 0.1; units=64
    drop = 0.1;
    # units = 32（ok）
    # units = 64  # _3 RLFM
    units = 32 # _4 没有气象变量 # 单独预测的NEE GPP RECO
    # y_SCALE = 0.1
    y_SCALE = 1
    # drop = 0.001;
    # 依次遍历划分好的训练集和测试集
    # ik = 0; kth_data = cross_validation_data[ik]
    for ik, kth_data in enumerate(cross_validation_data):
        # index_train, index_test, trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, RF_trainy, RF_testy = cross_validation_data
        trainx, trainy, testx, testy, train_covariate, test_covariate, train_sites_counts, test_sites_counts, test_sites_array = (
            kth_data['trainx'], kth_data['trainy'], kth_data['testx'], kth_data['testy'],
            kth_data['train_covariate'], kth_data['test_covariate'], kth_data['train_sites_counts'],
            kth_data['test_sites_counts'], kth_data['test_array'])

        trainy = trainy[:, :, :].reshape(trainy.shape[0], trainy.shape[1], 3)
        trainy[trainy != -9999.0] = trainy[trainy != -9999.0] * y_SCALE  # added by Hank
        testy = testy[:, :, :].reshape(testy.shape[0], testy.shape[1], 3)
        # 单个变量做训练
        trainy = trainy[:,:,:]
        testy = testy[:,:,:]
        # ## 对训练集和测试集的影像数据和其他协变量进行标准化
        input_images_train_norm0, input_images_test_norm0, input_covariate_train_norm0, input_covariate_test_norm0, mean_train, std_train, mean_train_cov, std_train_cov = Data_preprocess.construct_composite_train_test(
            trainx, testx, train_covariate, test_covariate, is_train_test_com=True, is_single_norm=True)

        # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
        # input_images_train_norm0 = other_process.process_pixel_values(input_images_train_norm0)
        # input_images_test_norm0 = other_process.process_pixel_values(input_images_test_norm0)
        # 仅选取MODIS 太阳辐射 和 空气温度
        input_images_train_norm0 = other_process.process_pixel_values(input_images_train_norm0[:,:,:,:,:])
        input_images_test_norm0 = other_process.process_pixel_values(input_images_test_norm0[:,:,:,:,:])
        # 复制数据 变换维度
        input_images_train_norm3 = np.transpose(input_images_train_norm0, (0, 1, 3, 4, 2))
        input_images_test_norm3 = np.transpose(input_images_test_norm0, (0, 1, 3, 4, 2))
        input_images_pre_norm3 = input_images_test_norm3
        train_n = input_images_train_norm3.shape[0]
        per_epoch = train_n // BATCH_SIZE
        start = time.time()
        # model = Model_build.get_transformer_cnn(layer_cnn=2, layern=2, units=units,
        # n_times=input_images_train_norm3.shape[1],
        # n_bands=input_images_train_norm3.shape[4], n_window=3, n_head=4, drop=drop, n_out=3,
        # n_features=14, is_batch=True, mask_value=-9999,
        # is_zero_fill=True, doy_encoder=doy_encoder,
        # is_rptv=False, is_sensor=False)
        importlib.reload(Model_build)
        # model = Model_build.get_transformer_cnn2(layer_cnn=1, layern=2, units=units, n_times=input_images_train_norm3.shape[1],
        #                 n_window=3,n_head=4, drop=drop,n_out=3, is_batch=True, mask_value=-9999.0, is_rptv=False,is_sensor=False)
        # (layer_cnn=1, layern=2, units=64, n_times=80,MODIS_BANDS_N = 9, METER_BANDS_N = 6, n_window=3, n_head=4, drop=0.1, n_out=1,is_batch=True, mask_value=-9999.0, is_rptv=False, is_sensor=False)

        model = Model_build.get_transformer_cnn2(layer_cnn=0, layern=3, units=units,
                                                 n_times=input_images_train_norm3.shape[1],
                                                 n_window=3, n_head=4, drop=drop, n_out=3, is_batch=True,
                                                 mask_value=-9999.0, is_rptv=True, is_sensor=False)
        if ik == 0:
            print(model.summary())  # 输出模型各层的参数状况
        print("**************************process" + str(ik + 1) + "th*********************************")
        input_images_train_norm3 = np.array(input_images_train_norm3)
        input_covariate_train_norm3 = np.array(input_covariate_train_norm0)
        # model_history = Model_history.my_train_1schedule(model, input_images_train_norm3[:, :, :, :, :],
        # trainy,
        # epochs=EPOCH, start_rate=LEARNING_RATE,
        # loss=loss, per_epoch=per_epoch, split_epoch=5,
        # option=METHOD, decay=L2, batch_size=BATCH_SIZE,
        # validation_split=0.04, hold_epoch=0, reduce_epoch=False)
        model_history = Model_history.my_train_1schedule_time_drop(model, input_images_train_norm3, trainy,
                                                                   epochs=EPOCH, start_rate=LEARNING_RATE, loss=loss,
                                                                   per_epoch=per_epoch, split_epoch=5,
                                                                   option=METHOD, decay=L2, batch_size=BATCH_SIZE,
                                                                   validation_split=0.04, hold_epoch=0,
                                                                   reduce_epoch=False, gaps_p=0.1)

        # 每一折都保存模型，模型名后面加第几折的后缀
        # 保存当前模型，模型名后面加上 epoch 编号
        # model_pre_name = f'./region model/get_cnn_transformer_RLM_epoch_{ik + 1}.h5'
        # model.save(model_pre_name)
        input_images_test_norm3 = np.array(input_images_test_norm3)
        input_covariate_test_norm3 = np.array(input_covariate_test_norm0)
        input_images_test_norm3 = tf.convert_to_tensor(input_images_test_norm3, dtype=tf.float32)
        input_covariate_test_norm3 = tf.convert_to_tensor(input_covariate_test_norm3, dtype=tf.float32)
        # test_datasetx = [input_images_test_norm3, input_covariate_test_norm3]
        # y_pred_filtered = model.predict(test_datasetx, verbose=2) / y_SCALE
        y_pred_filtered = model.predict(input_images_test_norm3, verbose=2) / y_SCALE

        # 将testy转换为float类型
        y_test_filtered = np.array(testy, dtype='float')
        Y_test_filtered.append(y_test_filtered)
        Y_pred_filtered.append(y_pred_filtered)
        Y_test_sites.append(test_sites_array)

    Y_pred_filtered_array = np.concatenate(Y_pred_filtered, axis=0)
    Y_test_filtered_array = np.concatenate(Y_test_filtered, axis=0)
    Y_test_sites_array = np.concatenate(Y_test_sites, axis=0)
    np.save(results_variation_path + 'DL_RLM_predict_MAE_1.npy', Y_pred_filtered_array)
    np.save(results_variation_path + 'DL_RLM_true_MAE_1.npy', Y_test_filtered_array)
    np.save(results_variation_path + 'DL_RLM_sites_MAE_1.npy', Y_test_sites_array)
