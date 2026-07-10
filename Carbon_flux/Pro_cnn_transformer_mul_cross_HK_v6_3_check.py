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
    y_image_array[:, :, 7]
    x_inter_mask_array[203, :, :, 0, 1]

    # ********************************
    # 检查匹配之后有x无y的情况
import numpy as np


# 假设已经加载了 x_image_array 和 y_image_array
# x_image_array.shape = (1688, 365, 15, 3, 3)
# y_image_array.shape = (1688, 365, 23)

def count_valid_pairs(x_data, y_data):
    """
    统计同时满足以下条件的数据点数量：
    1. x_data中每个位置（i,j）对应的15×3×3像素全部不为-9999
    2. y_data中每个位置（i,j）的第5、6、7个特征（索引4,5,6）全部为-9999
    """
    # 生成x的布尔掩码（检查所有15×3×3像素是否有效）
    mask_x = np.all(x_data != -9999, axis=(2, 3, 4))  # 结果形状 (1688, 365)

    # 生成y的布尔掩码（检查特征5-7是否全为无效值）
    # 注意：如果用户定义的索引从1开始，则y_data[:, :, [4,5,6]]对应第5、6、7个特征
    mask_y = np.any(y_data[:, :, [6,7, 8]] != -9999, axis=2)  # 形状 (1688, 365)

    # 统计同时满足两个条件的数据量
    valid_count = np.sum(mask_x & mask_y)

    return valid_count

# 执行统计
# result = count_valid_pairs(x_image_array_all,y_image_array_all)
result = count_valid_pairs(x_image_array,y_image_array)
print(f"满足条件的数据点数量为: {result}")
# 判断 x_image_array 中 15 个波段3x3窗口的像素值不全为 -9999 的个数
valid_x_count = np.sum(
    np.all(x_image_array != -9999, axis=(2, 3, 4))
)

# 判断 y_image_array 中的 (6:7) 区间不等于 -9999 的个数
# 这里假设你实际是想计算第6到第7个索引之间的数据，y的第三维的6和7是[6:7]不包含7
# 在这里我们会计算 y 的形状为(1788, 365, 6)
valid_y_count = np.sum(y_image_array[:, :, 6] != -9999)

# 输出结果
print("x_image_array中15个波段在3x3窗口的像素值不全为-9999的个数:", valid_x_count)
print("y_image_array中(6:7)不等于-9999的个数:", valid_y_count)