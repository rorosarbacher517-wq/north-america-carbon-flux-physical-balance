# v5_4 same as !v5_1! but using 64 + 0+4 
# v5_3 same as !v5_1! but using 64 + 0+3 
# !!v5_2!! same as !v5_1! but using 64 - quite good 
# v5_1 same as v!4_8! but using drop in training - drop help little 
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
    BATCH_SIZE = 16  # 16(OK),32,64,128,8
    L2 = 0.0001  # 0.0001(ok),0.00001,0.001,0.01
    
    EPOCH           =   int(sys.argv[1] )
    LEARNING_RATE   = float(sys.argv[2])
    BATCH_SIZE      =   int(sys.argv[3])
    L2              = float(sys.argv[4])
    
    # download the data
    base_name = base_name+".epoch{:03d}.rate{:.5f}.batch{:03d}.L{:.5f}".format(EPOCH,LEARNING_RATE,BATCH_SIZE,L2)
    inputdata_path = './data/sites_images_input/'
    x_ref_array = np.load(inputdata_path + 'x_ref_modis_interpolation_mark.npy', allow_pickle=True) # the dimension 15 indicate whether to interpolate 
    # 15 bands: 0-6 reflectance; 7-13 QA; 14 is interpolation indication (0 is measrued and 1 is interpolated)
    x_lai_array = np.load(inputdata_path + 'x_lai_modis_interpolation_mark.npy', allow_pickle=True)  # 4 bands: 2 bands for LAI, fpar; 1 band QA; 1 band interpolation indication 
    x_mete_array = np.load(inputdata_path + 'x_mete_era5_interpolation_mark.npy', allow_pickle=True) # ERA 5 meteorological 7 bands; 0-5 meteorological; 6  interpolation indication
    y_image_array_all = np.load(inputdata_path + 'y_flux_meteorological_mark.npy', allow_pickle=True)
    y_qa_array_all = np.load(inputdata_path + 'All_sites_y_qa.npy', allow_pickle=True)
    x_image_array_all = np.concatenate(
        (x_ref_array[..., :-1, :, :], x_lai_array[..., :-1, :, :], x_mete_array[..., :-1, :, :]), axis=2) # 23 bands all but not including interpolation indication
    x_inter_mask_all = np.concatenate(
        (x_ref_array[..., -1:, :, :], x_lai_array[..., -1:, :, :], x_mete_array[..., -1:, :, :]), axis=2) # 3 interpolation indication
    ##### 匹配前检查同一像素不同波段值缺失的情况
    # 将y_image_array_all进行特征填充
    # x_image_array is 7 reflectance + 2 lai/fpar + 6 ERA5 ()
    x_image_array, y_image_array, x_inter_mask_array0,y_qa_array = Data_filter_match.mul_match_x_y_modis(x_image_array_all,
                                                                                             y_image_array_all,
                                                                                             x_inter_mask_all,
                                                                                             y_qa_array_all,
                                                                                             ref=True, lai=True,
                                                                                             meteorology=False, # flux site meteorology 
                                                                                             era5_meteorology=True, # era5 meteorology 
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
    y_flux_qa_array = y_qa_array[:,:,-5:]
    # 将交叉验证的结果保存下来进行验证分析
    results_variation_path = './data/sites_estimate_result/'
    if not os.path.isdir(results_variation_path):
        # 创建文件夹
        os.makedirs(results_variation_path)
        print(f"已创建文件夹：{results_variation_path}")
    
    np.save(results_variation_path + 'DL_RLM_x_mask1.npy', x_inter_mask_array)
    np.save(results_variation_path + 'DL_RLM_y_qa1.npy', y_flux_qa_array) # y_flux_qa_array.shape (1688, 365, 5), the fouth dimension is QA 
    # 将质量标识的波段去除
    # x_image_array = x_image_array[:,:,[0,1,2,3,4,5,6,14,16,17,18,19,20,21,22],:,:]
    other_process.check_dim_fill(x_image_array)
    # ******************************************
    # check_dim_fill(x_image_array)
    # 对不符合条件的噪声值进行预处理
    # covariate_array = y_image_array[:, :, [3, 8, 9, 10, 11, 1, 2, 4, 15, 16, 17, 18, 19, 20, 21]]
    covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12,3,5,16,17,18, 19, 20, 21,22]]
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
    # k倍交叉验证划分数据集
    cross_validation_data = Data_preprocess.get_cross_validation_training_test(x_image_array, y_image_array, covariate_array)
    # print(cross_validation_data) # # 统计输入模型的数据-9999所占的比例
    # loss = mask_loss(maskValue=-9999)
    importlib.reload(myloss)
    loss = myloss.mask_loss_physical(maskValue=-9999)
    # loss = mask_loss_huber(maskValue=-9999)
    # loss = mask_loss_physical_MAE(maskValue=-9999)
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
    units = 64
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
        # ## 对训练集和测试集的影像数据和其他协变量进行标准化
        input_images_train_norm0, input_images_test_norm0, input_covariate_train_norm0, input_covariate_test_norm0, mean_train, std_train, mean_train_cov, std_train_cov = Data_preprocess.construct_composite_train_test(
            trainx, testx, train_covariate, test_covariate, is_train_test_com=True, is_single_norm=True)
        
        # # # 对输入的像素数据进行填充值处理 全为-9999的保持为-9999，部分为-9999的填为0
        input_images_train_norm0 = other_process.process_pixel_values(input_images_train_norm0)
        input_images_test_norm0  = other_process.process_pixel_values(input_images_test_norm0)
        # 复制数据 变换维度
        input_images_train_norm3 = np.transpose(input_images_train_norm0, (0, 1, 3, 4, 2))
        input_images_test_norm3 = np.transpose(input_images_test_norm0, (0, 1, 3, 4, 2))
        input_images_pre_norm3 = input_images_test_norm3
        train_n = input_images_train_norm3.shape[0]
        per_epoch = train_n // BATCH_SIZE
        start = time.time()
        
        importlib.reload(Model_build)        
        model = Model_build.get_transformer_cnn2(layer_cnn=1, layern=2, units=units, n_times=input_images_train_norm3.shape[1],
                        n_window=3,n_head=4, drop=drop,n_out=3, is_batch=True, mask_value=-9999.0, is_rptv=True,is_sensor=False)
        
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
        
        model_history = Model_history.my_train_1schedule_time_drop(model,input_images_train_norm3, trainy,epochs=EPOCH,start_rate=LEARNING_RATE,loss=loss,per_epoch=per_epoch,split_epoch=5,
                option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=0.04,hold_epoch=0,reduce_epoch=False, gaps_p=0.1)
        
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
    np.save(results_variation_path + 'DL_RLM_predict1.npy', Y_pred_filtered_array)
    np.save(results_variation_path + 'DL_RLM_true1.npy', Y_test_filtered_array)
    np.save(results_variation_path + 'DL_RLM_sites1.npy', Y_test_sites_array)
    DL_Y_predict = Y_pred_filtered_array
    DL_Y_true = Y_test_filtered_array
    DL_All_true = Y_test_sites_array
    inter_mask = x_inter_mask_array
    DL_y_qa = y_flux_qa_array
    importlib.reload(other_process) 
    # # x:插值与不插值；y:good quality(聚合后的Y_QA均值在0-1范围内) 和other
    DL_NEE_0, DL_GPP_0, DL_RECO_0, Ground_NEE_0, Ground_GPP_0, Ground_RECO_0, combined_mask_0 = other_process.interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, is_inter=0)
    
    # x:不插值；y:good quality 和other
    DL_NEE_1, DL_GPP_1, DL_RECO_1, Ground_NEE_1, Ground_GPP_1, Ground_RECO_1, combined_mask_1 = other_process.interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, is_inter=1)
    
    # # x:不插值；y:good quality
    DL_NEE_2, DL_GPP_2, DL_RECO_2, Ground_NEE_2, Ground_GPP_2, Ground_RECO_2, combined_mask_2 = other_process.interpolation_quality(
        DL_Y_predict, DL_Y_true, inter_mask, DL_y_qa, is_inter=2, threshold=0.2)
    
    X_variables = ['Ground_NEE_0', 'Ground_GPP_0', 'Ground_RECO_0', 'Ground_NEE_1', 'Ground_GPP_1', 'Ground_RECO_1','Ground_NEE_2', 'Ground_GPP_2', 'Ground_RECO_2']
    Y_variables = ['DL_NEE_0', 'DL_GPP_0', 'DL_RECO_0', 'DL_NEE_1', 'DL_GPP_1', 'DL_RECO_1', 'DL_NEE_2', 'DL_GPP_2','DL_RECO_2']
    # 创建一个2x3的子图，共6个散点密度子图
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # 绘制散点密度图并添加评估指标
    for i in range(9):
        row = i // 3
        col = i % 3
        x_variable = X_variables[i]
        y_variable = Y_variables[i]
        # Get the data for the current variable
        x = globals()[x_variable] # Get the data for the RF variable
        y = globals()[y_variable]  # Get the data for the DL variable
        # Calculate point density
        # xy = np.vstack([x, y]).astype(float)
        # z = gaussian_kde(xy)(xy)
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        # Normalize the density values to the range 0 to 1
        # z = (z - np.min(z)) / (np.max(z) - np.min(z))
        # Plot the scatter density plot and set the frequency (probability)
        # z[z<0.1] = 0 
        # scatter = axs[row, col].scatter(x, y, c=z, cmap='Spectral_r', s=20)
        cmap = plt.cm.Spectral_r
        cmap.set_under("white")
        from matplotlib.colors import LogNorm
        from matplotlib.colors import Normalize
        # density_threshold = 0.1
        density_threshold = 5
        h = axs[row, col].hist2d(x, y, bins=200, cmap=cmap, norm=Normalize(vmin=density_threshold))  # Adjust bins for density granularity
        # h = axs[row, col].hist2d(x, y, bins=100, cmap=cmap)  # Adjust bins for density granularity
        # plt.hist2d(x, y, bins=100, cmap=cmap)  # Adjust bins for density granularity
        # plt.colorbar(label='Density')
        # Set backgr
        # scatter = axs[row, col].scatter(x, y, c=z, cmap=cmap, s=20, vmin=0.1)
        # plt.colorbar(scatter, ax=axs[row, col], label='Density')
        # Add color bar to each subplot
        plt.colorbar(h[3], ax=axs[row, col], label='Density')
        min_val = min(np.min(y), np.min(y))
        max_val = max(np.max(y), np.max(y))
        axs[row, col].set_xlim([min_val, max_val])
        axs[row, col].set_ylim([min_val, max_val])
        axs[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        # Calculate RMSE, R2, and bias evaluation metrics
        rmse = np.sqrt(mean_squared_error(x, y))
        r2 = r2_score(x, y)
        n = len(x)
        bias = np.mean(x - y)
        cc = np.corrcoef(x, y)[0, 1]
        # 计算相对均方根误差（RRMSE）
        mean_value = np.mean(np.absolute(x) )
        rrmse = rmse / mean_value
        # Add text annotations
        text = "RMSE = {:.2f}\nRRMSE = {:.2f}\nCC = {:.2f}\nBias = {:.2f}\nn = {}".format(rmse, rrmse, cc, bias, n)
        axs[row, col].text(0.05, 0.95, text, verticalalignment='top', transform=axs[row, col].transAxes,
                           bbox=dict(facecolor='none', edgecolor='red'), color='red', fontweight='bold')
        
        axs[row, col].set_title(f"{Y_variables[i]} vs {X_variables[i]}")
        
        # break
    
    plt.suptitle('Comparison of predicted values and observed values')
    plt.tight_layout()
    plt.savefig(base_name+'plot.png')
    # plt.show()
