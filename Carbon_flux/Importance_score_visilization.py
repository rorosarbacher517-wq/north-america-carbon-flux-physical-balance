import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Data_filter_match

import myloss
import other_process
import importlib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
importlib.reload(myloss)
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
##### 匹配前检查同一像素不同波段值缺失的情况

# 将y_image_array_all进行特征填充
# x_image_array is 7 reflectance + 2 lai/fpar + 6 ERA5 ()
x_image_array0, y_image_array0, x_inter_mask_array0, y_qa_array = Data_filter_match.mul_match_x_y_modis_quality(
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




import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 加载模型
model_path = './region model/get_cnn_transformer_RLM.h5'
model = tf.keras.models.load_model(model_path)

Carbon_flux_array = y_image_array0[:,:,6:9] # shape 1688,365,3；3代表3个碳通量"NEE", "GPP", "RECO"
x_image_array_transfer = np.transpose(x_image_array0, (0, 1, 3, 4, 2)) # shape 1688,365,3,3,15 模型的数据输入

carbon_output = model.predict(x_image_array_transfer) # shape 为1688，365，3
print(carbon_output.shape)

#****************************************************
# # 可视化保存模型的重要性得分
import shap

def replace_nan_with_mean(arr, nan_val=-9999):
    mask = (arr == nan_val)
    arr_mean = np.nanmean(np.where(mask, np.nan, arr), axis=(0, 1, 2, 3), keepdims=True)
    return np.where(mask, arr_mean, arr)

x_image_array_transfer = replace_nan_with_mean(x_image_array_transfer, nan_val=-9999)
# 将
# 定义子模型函数
def create_sub_model(original_model, output_index):
    return tf.keras.Model(
        inputs=original_model.inputs,
        outputs=original_model.outputs[output_index]  # 输出形状 (batch_size, 365)
    )

sub_models = [
    create_sub_model(model, 0),  # NEE
    create_sub_model(model, 1),  # GPP
    create_sub_model(model, 2)   # RECO
]

# 随机选择50个样本
sample_indices = np.random.choice(x_image_array_transfer.shape[0], 50, replace=False)
x_sample = x_image_array_transfer[sample_indices]  # (50, 365, 3, 3, 15)

# 聚合时空维度：时间步和空间位置取均值
x_sample_agg = np.mean(x_sample, axis=(1, 2, 3))  # (50, 15)

# 使用KernelExplainer适配任意模型
explainer = shap.KernelExplainer(
    model=sub_models[0].predict,  # 以NEE为例
    data=x_sample_agg[:10]        # 背景数据
)

# 计算SHAP值
shap_values = explainer.shap_values(x_sample_agg[:10])

target_names = ["NEE", "GPP", "RECO"]
feature_names = [f"Var {i+1}" for i in range(15)]
shap_importances = []

for idx, target in enumerate(target_names):
    explainer = shap.KernelExplainer(
        sub_models[idx].predict,
        x_sample_agg[:10]
    )
    shap_values = explainer.shap_values(x_sample_agg[:10])
    shap_importances.append(np.abs(shap_values).mean(axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (target, importance) in enumerate(zip(target_names, shap_importances)):
    axes[i].barh(feature_names, importance, color='#4c72b0')
    axes[i].set_title(f"{target} Feature Importance")
    axes[i].set_xlabel('Mean |SHAP Value|')
    axes[i].invert_yaxis()

plt.tight_layout()
plt.show()
# #****************************************************
# # 可视化预测变量与目标变量的相关性系数 热力图
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # ===================== 1. 处理 x 和 y 数据 =====================
# # 提取中心像元数据（去掉 3x3 维度）
# x_processed = x_image_array0[..., 1, 1]  # 形状变为 (1688, 365, 15)
# y_processed = Carbon_flux_array  # 形状为 (1688, 365, 3)
#
# # 展平成 (1688*365, 15) 和 (1688*365, 3)
# x_processed = x_processed.reshape(-1, 15)
# y_processed = y_processed.reshape(-1, 3)
#
# # ===================== 2. 处理缺失值 (-9999 替换为 NaN) =====================
# x_processed[x_processed == -9999] = np.nan
# y_processed[y_processed == -9999] = np.nan
#
# # ===================== 3. 转换为 DataFrame =====================
# predictor_names = [
#     'Band1', 'Band2', 'Band3', 'Band4',
#     'Band5', 'Band6', 'Band7',
#     'LAI', 'FAPAR',
#     'Solar radiation', 'Air temperature',
#     'Soil water L1', 'Soil water L2', 'Soil water L3', 'Soil water L4'
# ]
#
# target_names = ['NEE', 'GPP', 'RECO']
#
# # 创建 DataFrame
# df = pd.DataFrame(
#     np.hstack([x_processed, y_processed]),  # 合并 x 和 y
#     columns=predictor_names + target_names
# )
#
# # ===================== 4. 计算相关性矩阵 =====================
# # 计算 Pearson 相关性（自动忽略 NaN）
# corr_matrix_full = df.corr()
#
# # 提取 15 个预测变量 与 3 个目标变量 之间的相关性
# corr_matrix = corr_matrix_full.loc[predictor_names, target_names].to_numpy()
#
# # ===================== 5. 可视化相关性矩阵 =====================
# plt.rcParams['font.family'] = 'Arial'
# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap='RdBu_r',  # 强调正负相关
#     center=0,
#     vmin=-1, vmax=1,
#     linewidths=0.5,
#     xticklabels=target_names,
#     yticklabels=predictor_names,
#     annot_kws={'size': 16},
#     cbar_kws={'label': 'Pearson correlation coefficient'}
# )
#
# # 美化图表
# # plt.title('Correlation Matrix between Predictors and Carbon Flux Components', fontsize=16, fontweight='bold', pad=20)
# plt.xlabel('Carbon Fluxes', fontsize=18, labelpad=15)
# plt.ylabel('Predictor Variables', fontsize=18, labelpad=15)
# plt.xticks(rotation=45, ha='right', fontsize=16)
# plt.yticks(rotation=0, fontsize=16)
#
# # 添加网格线增强可读性
# ax.hlines(range(1, 15), *ax.get_xlim(), colors='gray', linewidth=0.3, linestyle='--')
# ax.vlines(range(1, 3), *ax.get_ylim(), colors='gray', linewidth=0.3, linestyle='--')
# # 设置颜色条标签字体
# cbar = ax.collections[0].colorbar
# cbar.ax.yaxis.label.set_fontsize(18)
# cbar.ax.tick_params(labelsize=16)  # 设置颜色条刻度字体
#
# plt.tight_layout()
# resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
# plt.savefig(resluts_path + 'Pearson Correlation' + '.png', dpi=600)
# plt.show()