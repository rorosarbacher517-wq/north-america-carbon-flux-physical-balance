import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model
model_path = './region model/get_cnn_transformer_RLM.h5'
model = tf.keras.models.load_model(model_path)

input_data_path = './data/sites_results/'
x_image_array = np.load(input_data_path + 'DL_RLM_x_image_array.npy', allow_pickle=True)
y_image_array = np.load(input_data_path + 'DL_RLM_y_image_array.npy', allow_pickle=True)
covariate_array = np.load(input_data_path + 'DL_RLM_covariate_array.npy', allow_pickle=True)

x_image_array_transfer = np.transpose(x_image_array, (0, 1, 3, 4, 2))

#****************************************************
# # 显示某个特定层的注意力权重 365天
# Define attention layers to extract weights
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model
model_path = './region model/get_cnn_transformer_RLM.h5'
model = tf.keras.models.load_model(model_path)

input_data_path = './data/sites_results/'
x_image_array = np.load(input_data_path + 'DL_RLM_x_image_array.npy', allow_pickle=True)
y_image_array = np.load(input_data_path + 'DL_RLM_y_image_array.npy', allow_pickle=True)
covariate_array = np.load(input_data_path + 'DL_RLM_covariate_array.npy', allow_pickle=True)

x_image_array_transfer = np.transpose(x_image_array, (0, 1, 3, 4, 2))

#****************************************************
# # # 显示单层的注意力权重
# Define attention layers to extract weights
attention_layers_names = ['tf.concat_3']  # Only keep the 'tf.concat_3' layer
attention_weights_list = []

# Extract weights from the attention layer
for layer_name in attention_layers_names:
    attention_layer = model.get_layer(layer_name)
    attention_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.output)
    # attention_model = tf.keras.Model(
    #     inputs=model.input,
    #     outputs=attention_layer.attention_scores  # 注意力分数形状：(batch, num_heads, seq_len, seq_len)
    # )
    # Predict using actual input data shape 1688,365,3
    attention_weights = attention_model.predict(x_image_array_transfer)

    # Average the weights along the sample and last dimension
    # attention_weights_carbon = attention_weights[:,:,2]
    # average_attention_weight = np.mean(attention_weights_carbon, axis=0)
    average_attention_weight = np.mean(attention_weights, axis=(0,2))
    # Append results to list
    attention_weights_list.append(average_attention_weight)

# Convert to NumPy array and reshape to (365, 1)
average_attention_all = np.array(attention_weights_list).T

# Calculate averages for specified ranges
mean_90_to_270 = np.mean(average_attention_all[90:271], axis=0)  # Days 90-270
mean_1_to_89 = np.mean(average_attention_all[0:90], axis=0)      # Days 1-89
mean_271_to_365 = np.mean(average_attention_all[271:], axis=0)   # Days 271-365

# Combine means for the 1-89 and 271-365 ranges
mean_combined = (mean_1_to_89 + mean_271_to_365) / 2

# Create a single subplot
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.1)  # 增加底部边距
# Set color for scientific publication
line_color = '#0F3D5E'    # IBM蓝色系
fill_color1 = '#F4B183'    # 橙色填充 (90-270)
fill_color2 = '#D9D9D9'    # 灰色填充 (其他区域)



# Plot average attention weights for the head
ax.plot(np.arange(365), average_attention_all[:, 0],color=line_color,
        label='Attention Weight')

# Add mean range lines with defined colors separately
ax.axvspan(90, 270, facecolor=fill_color1, alpha=0.5,label='Growing season(GS)')  # Shade for 90-270
ax.axvspan(0, 89, facecolor=fill_color2, alpha=0.5,label='Non-growing season (NGS)')  # Shade for 1-89
ax.axvspan(271, 364, facecolor=fill_color2, alpha=0.5,label='Non-growing season (NGS)')  # Shade for 271-365

ax.text(45, 0.8*ax.get_ylim()[1],f'Non-growing\nseason (NGS)',ha='center', color='black',alpha=0.5,fontsize=14)
ax.text(180, 0.8*ax.get_ylim()[1],f'Growing season(GS)',ha='center', color='black',alpha=0.5,fontsize=14)
ax.text(315, 0.8*ax.get_ylim()[1],f'Non-growing\nseason (NGS)',ha='center', color='black',alpha=0.5,fontsize=14)

# ax.set_title(attention_layers_names[0], fontsize=14, fontname='Arial')  # Set title font
ax.set_ylabel('Attention Weight', fontsize=16, fontname='Arial', labelpad=10)       # Label font

# Create a modified legend
ax.legend(['Attention Weight',
            f'Mean of GS: {mean_90_to_270[0]:.2f} ± {np.std(average_attention_all[90:271]):.2f}',
            f'Mean of NGS: {mean_combined[0]:.2f} ± {np.std(np.concatenate([average_attention_all[0:90], average_attention_all[271:365]])):.2f}'],
           fontsize=12,loc='upper right')

# Set X axis label
ax.set_xlabel('Day of year (DOY)', fontsize=16, fontname='Arial',labelpad=15)  # Change 'Days' to 'DOY'
ax.set_xticks(np.arange(0, 365, step=30))  # Every 30 days marked

# Title for the entire figure
fig.suptitle('Average Attention Weights of Multi-Head Attention in Time Series', fontsize=18, fontname='Arial',y=0.95)

plt.rcParams.update({
    'xtick.labelsize': 14,  # X轴刻度字体大小
    'ytick.labelsize': 14   # Y轴刻度字体大小
})
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the title rect=[0, 0.03, 1, 0.90]

resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Attention weight/'
plt.savefig(resluts_path + 'Avearage MHA of 365 days_v1' + '.png', dpi=300)
plt.show()
#****************************************************
# # 显示多层的注意力权重
# # Define attention layers to extract weights
# attention_layers_names = ['tf.concat_3','dense_25']
# attention_weights_list = []
#
# # Extract weights from each layer
# for layer_name in attention_layers_names:
#     attention_layer = model.get_layer(layer_name)
#     attention_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.output)
#
#     # Predict using actual input data
#     attention_weights = attention_model.predict(x_image_array_transfer)
#
#     # Average the weights along the sample and last dimension
#     average_attention_weight = np.mean(attention_weights, axis=(0, 2))
#
#     # Append results to list
#     attention_weights_list.append(average_attention_weight)
#
# # Convert to NumPy array and reshape to (365, 2)
# average_attention_all = np.array(attention_weights_list).T
#
# # Calculate averages for specified ranges
# mean_90_to_270 = np.mean(average_attention_all[90:271], axis=0)
# mean_1_to_89 = np.mean(average_attention_all[0:90], axis=0)  # Days 1-89
# mean_271_to_365 = np.mean(average_attention_all[271:], axis=0)  # Days 271-365
#
# # Combine means for the 1-89 and 271-365 ranges
# mean_combined = (mean_1_to_89 + mean_271_to_365) / 2
#
# # Create subplots
# n_heads = average_attention_all.shape[1]
# fig, axs = plt.subplots(n_heads, 1, figsize=(12, 6), sharex=True)
#
# # Set colors for scientific publication
# colors = ['blue', 'green']
#
# # Plot average attention weights for each head
# for i in range(n_heads):
#     axs[i].plot(np.arange(365), average_attention_all[:, i], marker='o', linestyle='-', color=colors[i],
#                 label='Attention Weight')
#
#     # Add mean range lines with defined colors separately
#     axs[i].axvspan(90, 270, color='lightcoral', alpha=0.5)  # Shade for 90-270
#     # axs[i].text(180, max(average_attention_all[:, i]),
#     #             f'Mean (90-270): {mean_90_to_270[i]:.2f}',
#     #             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')
#
#     # Shade for 1-89
#     axs[i].axvspan(0, 89, color='lightgrey', alpha=0.3)
#     # axs[i].text(45, max(average_attention_all[:, i]) * 0.9,
#     #             f'Mean (1-89): {mean_1_to_89[i]:.2f}',
#     #             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')
#
#     # Shade for 271-365
#     axs[i].axvspan(271, 364, color='lightgrey', alpha=0.3)
#     # axs[i].text(316, max(average_attention_all[:, i]) * 0.9,
#     #             f'Mean (271-365): {mean_271_to_365[i]:.2f}',
#     #             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')
#
#     axs[i].set_title(attention_layers_names[i])
#     axs[i].set_ylabel('Attention Weight')
#     axs[i].grid()
#
# # Create a modified legend for each subplot
#     axs[i].legend(['Attention Weight',
#                 f'Mean (90-270): {mean_90_to_270[0]:.2f}',
#                 f'Mean (1-89 & 271-365): {mean_combined[0]:.2f}'],
#                 loc='upper right')
#
# # Set X axis label
# axs[-1].set_xlabel('Days')
# axs[-1].set_xticks(np.arange(0, 365, step=30))  # Every 30 days marked
#
# # # Create a modified legend to include the two mean ranges
# # axs[0].legend(['Attention Weight',
# #                 f'Mean (90-270): {mean_90_to_270[0]:.2f}',
# #                 f'Mean (1-89 & 271-365): {mean_combined[0]:.2f}'],
# #                 loc='upper right')
#
#
# plt.tight_layout()  # Adjust subplots to avoid overlap
# plt.show()

# # 打印四个头一年365天的注意力权重
#
# # # 加载模型
# model_path = './region model/get_cnn_transformer_RLM.h5'
# model = tf.keras.models.load_model(model_path)
#
# input_data_path = './data/sites_results/'
# x_image_array = np.load(input_data_path + 'DL_RLM_x_image_array.npy',allow_pickle=True)
# y_image_array = np.load(input_data_path + 'DL_RLM_y_image_array.npy',allow_pickle=True)
# covariate_array = np.load(input_data_path + 'DL_RLM_covariate_array.npy',allow_pickle=True)
# x_image_array_transfer = np.transpose(x_image_array, (0, 1, 3, 4, 2))
# # # 定义需要提取注意力权重的层
# attention_layers_names = ['multi_head_attention_4', 'multi_head_attention_5',
#                           'multi_head_attention_6', 'multi_head_attention_7']
# # attention_layers_names = ['dense_25','tf.concat_3']
# # 存储每层的注意力权重
# attention_weights_list = []
#
# # 提取每一层的权重
# for layer_name in attention_layers_names:
#     attention_layer = model.get_layer(layer_name)
#     attention_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.output)
#
#     # 使用实际的输入数据进行预测
#     attention_weights = attention_model.predict(x_image_array_transfer)
#
#     # 平均计算沿样本和最后一个维度
#     average_attention_weight = np.mean(attention_weights, axis=(0, 2))  # 结果形状为 (365, n_heads)
#
#     # 将结果添加到列表中
#     attention_weights_list.append(average_attention_weight)
#
# # 转换为 NumPy 数组，形状为 (4, 365)，转置为 (365, 4)
# average_attention_all = np.array(attention_weights_list).T  # 这里我们转置以便后续绘图
#
# # 创建子图
# n_heads = average_attention_all.shape[1]  # 获取头数
# fig, axs = plt.subplots(n_heads, 1, figsize=(12, 6), sharex=True)
#
# # 为每个头绘制平均注意力权重
# for i in range(n_heads):
#     axs[i].plot(np.arange(365), average_attention_all[:, i], marker='o', linestyle='-', color='b')
#     axs[i].set_title(attention_layers_names[i])
#     axs[i].set_ylabel('Attention Weight')
#     axs[i].grid()
#
# # 设置 X 轴标签
# axs[-1].set_xlabel('Days')
# axs[-1].set_xticks(np.arange(0, 365, step=30))  # 每30天标一个刻度
#
# plt.tight_layout()  # 调整子图以避免重叠
# plt.show()
#****************************************************
# # 打印4个头 最后每个维度的注意力权重
# # 加载模型
# model_path = './region model/get_cnn_transformer_RLM.h5'
# model = tf.keras.models.load_model(model_path)
#
# # 打印每个多头注意力层的第六个权重
# attention_layers = [model.get_layer(f'multi_head_attention_{i + 4}') for i in
#                     range(4)]  # 这里有multi_head_attention_4到 multi_head_attention_7
# weights_list = []
#
# for i, attention_layer in enumerate(attention_layers):
#     weights = attention_layer.get_weights()  # 获取权重
#     if len(weights) > 6:  # 检查是否有第六个权重
#         weight6 = weights[6]  # 获取第六个权重
#         print(f"Attention Layer: multi_head_attention_{i + 4}, Weight 6 shape: {weight6.shape}")
#         weights_list.append(weight6)  # 将第六个权重添加到列表中
#     else:
#         print(f"Attention Layer: multi_head_attention_{i + 4} does not have a sixth weight.")
#
# # 四个头的注意力权重的堆叠图
# # 变换为 NumPy 数组
# weights_array = np.array(weights_list)  # 假设 weights_list 的形状是 (4, 16)
#
# # Calculate the average attention weights across the last dimension
# mean_weights = np.mean(weights_array, axis=-1)  # Resulting shape will be (4, 4, 16)
#
# # mean_weights now has shape (n_heads, n_features, n_dimensions) => (4, 4, 16)
# # For visualization, we can average over the features across the heads
# average_feature_weights = np.mean(mean_weights, axis=0)  # Resulting shape will be (4, 16)
#
# # Plot the results
# n_heads = average_feature_weights.shape[0]  # Number of heads, should be 4
# x = np.arange(average_feature_weights.shape[1])  # x represents dimension indices, (0 to 15)
#
# #############***********
# # 对每个特征维度再次取平均，得到 16 个特征的总体平均权重
# final_average_weights = np.mean(average_feature_weights, axis=0)  # 结果形状应为 (16,)
#
# # 设置 x 轴数据，代表不同特征维度
# x = np.arange(final_average_weights.shape[0])  # x 代表特征索引 (0 到 15)
#
# # 可视化平均注意力权重
# plt.figure(figsize=(12, 6))
#
# # 绘制柱形图
# plt.bar(x, final_average_weights, color='skyblue')
#
# # 图形配置
# plt.title('Average Attention Weights Across All Heads for Each Dimension')
# plt.xlabel('Feature Dimensions')
# plt.ylabel('Average Attention Weight')
# plt.xticks(x, [f'Dim {j + 1}' for j in x])  # 特征维度标记
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()
#
# ######**************************************
#
# plt.figure(figsize=(12, 6))
#
# # Stacked Bar Plot
# bottom_values = np.zeros(average_feature_weights.shape[1])  # Initialize the bottom values for stacking
# for i in range(n_heads):
#     plt.bar(x, average_feature_weights[i], bottom=bottom_values, label=f'Head {i + 1}')
#     bottom_values += average_feature_weights[i]  # Update the bottom values
#
# # Configure the plot
# plt.title('Stacked Average Attention Weights for Each Head Across Feature Dimensions')
# plt.xlabel('Feature Dimensions')
# plt.ylabel('Average Attention Weight')
# plt.xticks(x, [f'Dim {j + 1}' for j in x])  # Dimension labels
# plt.legend()
# plt.grid(axis='y')
# plt.tight_layout()  # Adjust the layout
# plt.show()
# #****************************************************
# # 打印15个特征变量的注意力权重
# import tensorflow as tf
# import numpy as np
#
# # 查找 'time_distributed_1' 层
# time_distributed_layer = None
# for layer in model.layers:
#     if layer.name == 'time_distributed_1':
#         time_distributed_layer = layer
#         break
#
# # 加载模型
# model_path = './region model/get_cnn_transformer_RLM.h5'
# model = tf.keras.models.load_model(model_path)
#
# # 查找 'time_distributed_1' 层
# time_distributed_layer = None
# for layer in model.layers:
#     if layer.name == 'time_distributed_1':
#         time_distributed_layer = layer
#         break
#
# if time_distributed_layer:
#     print(f"\nFound 'time_distributed_1' layer. Extracting weights...\n")
#
#     # 获取该层的权重
#     kernel_weight = time_distributed_layer.get_weights()[0]  # (3, 3, 9, 64)
#
#     # 对 3x3 维度取均值，保留 9 个输入通道的信息
#     attention_weights_9 = np.mean(kernel_weight, axis=(0, 1, 3))  # 形状 (9,)
#
#     print("\nAttention weights for 9 input channels (averaged over 3x3 and output channels):")
#     print(attention_weights_9.shape)  # 应该是 (9,)
#     print(attention_weights_9)  # 打印9个通道的注意力权重值
#
# else:
#     print("Layer 'time_distributed_1' not found.")

#****************************************************
# # 打印3*3窗口的注意力权重
# import tensorflow as tf
# import numpy as np
#
# # 加载模型
# model_path = './region model/get_cnn_transformer_RLM_1.h5'
# model = tf.keras.models.load_model(model_path)
#
# # 查找 'time_distributed_1' 层
# time_distributed_layer = None
# for layer in model.layers:
#     if layer.name == 'time_distributed_1':
#         time_distributed_layer = layer
#         break
#
# if time_distributed_layer:
#     print(f"\nFound 'time_distributed_1' layer. Extracting weights...\n")
#
#     # 获取该层的权重
#     kernel_weight = time_distributed_layer.get_weights()[0]  # (3, 3, 9, 64)
#
#     # 计算最终 3x3 窗口的权重（对输入通道和输出通道取均值）
#     final_3x3_weight = np.mean(kernel_weight, axis=(2, 3))  # 形状 (3,3)
#
#     print("\nFinal 3x3 window weight (averaged across all channels):")
#     print(final_3x3_weight.shape)  # 应该是 (3,3)
#     print(final_3x3_weight)  # 打印最终的 3x3 权重值
#
# else:
#     print("Layer 'time_distributed_1' not found.")

#****************************************************
# # # 查看每一层的注意力权重
# import tensorflow as tf
# import numpy as np
#
# # 加载模型
# model_path = './region model/get_cnn_transformer_RLM_1.h5'
# model = tf.keras.models.load_model(model_path)
#
# # 打印模型结构
# model.summary()
#
# # 打印每层的权重
# for layer in model.layers:
#     print(f"Layer: {layer.name}")
#     if layer.weights:  # 如果层有权重
#         for weight in layer.weights:
#             print(f"  Weight name: {weight.name}")
#             print(f"  Weight shape: {weight.shape}")
#             # print(f"  Weight values: {weight.numpy()}")
#
# # 全局变量用于保存注意力权重
# attention_weights_list = []
#
# # 遍历模型的每一层
# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.MultiHeadAttention):
#         # 提取注意力权重
#         weights = layer.get_weights()
#         attention_weights_list.append(weights)
#         print(f"Attention weights from layer: {layer.name}")
#         for i, weight in enumerate(weights):
#             print(f"  Weight {i}: {weight.shape}")
#             # print(weight)  # 打印权重值

# 保存注意力权重
# np.save('./region model//attention_weights.npy', attention_weights_list)