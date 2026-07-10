import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model
model_path = './region model/get_cnn_transformer_RLM_1.h5'
model = tf.keras.models.load_model(model_path)


input_data_path = './data/sites_results/'
x_image_array = np.load(input_data_path + 'DL_RLM_x_image_array.npy', allow_pickle=True)
y_image_array = np.load(input_data_path + 'DL_RLM_y_image_array.npy', allow_pickle=True)
covariate_array = np.load(input_data_path + 'DL_RLM_covariate_array.npy', allow_pickle=True)
attention_array = np.load(input_data_path + 'DL_RLM_attention.npy', allow_pickle=True)


# x_image_array_transfer = np.transpose(x_image_array, (0, 1, 3, 4, 2))
# # 使用模型进行预测
# test_batch_size = 32
# test_n = x_image_array_transfer.shape[0]
# predictions_list, attentions_list = [], []
#
# for i in range(0, test_n, test_batch_size):
#     batch_x = x_image_array_transfer[i:i + test_batch_size]
#     result = model.predict(batch_x, verbose=0)
#     predictions_list.append(result['predictions'])
#     attentions_list.append(result['attentions'])
#
# # 拼接 predictions
# predictions = np.concatenate(predictions_list, axis=0)
# # 拼接 attentions
# attentions = np.concatenate(attentions_list, axis=1)  # axis=1是 batch 维度

# outputs = model.predict(x_image_array_transfer)
#
# # 从输出中提取 predictions 和 attentions
# predictions = outputs['predictions']
# attentions = outputs['attentions']
#
# # 打印 attentions 的形状
# print("Attentions shape:", attentions.shape)
#
# # 假设你想提取第一层的注意力分数
# layer_index = 0  # 第一层
# layer_attention = attentions[layer_index]
#
# # 打印特定层注意力分数的形状
# print(f"Layer {layer_index} attention shape:", layer_attention.shape)
#
# # 打印特定层的一些注意力分数
# print(f"Sample layer {layer_index} attention scores:", layer_attention[0])  # 第一个样本的注意力分数
# # # 打印模型结构
# model.summary()
#
# 打印每层的权重
for layer in model.layers:
    print(f"Layer: {layer.name}")
    if layer.weights:  # 如果层有权重
        for weight in layer.weights:
            print(f"  Weight name: {weight.name}")
            print(f"  Weight shape: {weight.shape}")
            # print(f"  Weight values: {weight.numpy()}")
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
#
# #****************************************************
# 打印3*3窗口的注意力权重
import tensorflow as tf
import numpy as np

# 查找 'time_distributed_1' 层
time_distributed_layer = None
for layer in model.layers:
    if layer.name == 'time_distributed_1':
        time_distributed_layer = layer
        break

# # 打印所有层的名称，以便找到包含注意力的层
# for layer in model.layers:
#     print(layer.name)

# attention_layer_name = 'time_distributed_1'  # 替换为实际的注意力层名称
# attention_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(attention_layer_name).output)
# # # 使用新模型进行预测，获取注意力分数
# attention_scores = attention_model.predict(x_image_array_transfer)

#
# # 打印注意力分数的形状
# print("Attention scores shape:", attention_scores.shape)

if time_distributed_layer:
    print(f"\nFound 'time_distributed_1' layer. Extracting weights...\n")

    # 获取该层的权重
    kernel_weight = time_distributed_layer.get_weights()[0]  # (3, 3, 9, 64)

    # 计算最终 3x3 窗口的权重（对输入通道和输出通道取均值）
    final_3x3_weight = np.mean(kernel_weight, axis=(2, 3))  # 形状 (3,3)

    print("\nFinal 3x3 window weight (averaged across all channels):")
    print(final_3x3_weight.shape)  # 应该是 (3,3)
    print(final_3x3_weight)  # 打印最终的 3x3 权重值
else:
    print("Layer 'time_distributed_1' not found.")
