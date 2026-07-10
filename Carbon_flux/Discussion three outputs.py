import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# Using RLFM
rmse_data = {
    ('NEE', 'Physics-aware transformer'): [1.10, 1.13, 1.11],
    ('NEE', 'XGBoost'): [1.14, 1.17, 1.15],
    ('NEE', 'RF'): [1.14, 1.16, 1.13],
    ('GPP', 'Physics-aware transformer'): [1.53, 1.56, 1.53],
    ('GPP', 'XGBoost'): [1.58, 1.66, 1.62],
    ('GPP', 'RF'): [1.57, 1.65, 1.63],
    ('RECO', 'Physics-aware transformer'): [1.30, 1.29, 1.27],
    ('RECO', 'XGBoost'): [1.34, 1.35, 1.32],
    ('RECO', 'RF'): [1.34, 1.33, 1.31]
}

cc_data = {
    ('NEE', 'Physics-aware transformer'): [0.53, 0.54, 0.54],
    ('NEE', 'XGBoost'): [0.47, 0.48, 0.48],
    ('NEE', 'RF'): [0.48, 0.49, 0.50],
    ('GPP', 'Physics-aware transformer'): [0.81, 0.81, 0.81],
    ('GPP', 'XGBoost'): [0.79, 0.77, 0.78],
    ('GPP', 'RF'): [0.79, 0.78, 0.77],
    ('RECO', 'Physics-aware transformer'): [0.79, 0.80, 0.79],
    ('RECO', 'XGBoost'): [0.77, 0.77, 0.76],
    ('RECO', 'RF'): [0.77, 0.77, 0.77]
}

# bias_data = {
#     ('NEE', 'CFR'): [-0.04, -0.01, 0.04],
#     ('NEE', 'XGBoost'): [-0.03, 0.00, 0.05],
#     ('NEE', 'RF'): [-0.04, 0.00, 0.06],
#     ('GPP', 'CFR'): [0.01, 0.01, -0.06],
#     ('GPP', 'XGBoost'): [-0.02, -0.03, -0.17],
#     ('GPP', 'RF'): [-0.06, -0.06, -0.20],
#     ('RECO', 'CFR'): [-0.02, 0.01, -0.02],
#     ('RECO', 'XGBoost'): [-0.05, 0.00, -0.07],
#     ('RECO', 'RF'): [-0.10, -0.06, -0.13]
# }
# 绝对值
bias_data = {
    ('NEE', 'Physics-aware transformer'): [0.04, 0.01, 0.04],
    ('NEE', 'XGBoost'): [0.03, 0.00, 0.05],
    ('NEE', 'RF'): [0.04, 0.00, 0.06],
    ('GPP', 'Physics-aware transformer'): [0.01, 0.01, 0.06],
    ('GPP', 'XGBoost'): [0.02, 0.03, 0.17],
    ('GPP', 'RF'): [0.06, 0.06, 0.20],
    ('RECO', 'Physics-aware transformer'): [0.02, 0.01, 0.02],
    ('RECO', 'XGBoost'): [0.05, 0.00, 0.07],
    ('RECO', 'RF'): [0.10, 0.06, 0.13]
}

# Using RLF
# Prepare the new data
# # RLF
# rmse_data = {
#     ('NEE', 'Physics-aware transformer'): [1.20, 1.22, 1.18],
#     ('NEE', 'XGBoost'): [1.22, 1.23, 1.20],
#     ('NEE', 'RF'): [1.24, 1.26, 1.22],
#     ('GPP', 'Physics-aware transformer'): [1.64, 1.67, 1.61],
#     ('GPP', 'XGBoost'): [1.69, 1.73, 1.68],
#     ('GPP', 'RF'): [1.71, 1.76, 1.71],
#     ('RECO', 'Physics-aware transformer'): [1.41, 1.42, 1.37],
#     ('RECO', 'XGBoost'): [1.43, 1.42, 1.38],
#     ('RECO', 'RF'): [1.45, 1.43, 1.39]
# }
#
# cc_data = {
#     ('NEE', 'Physics-aware transformer'): [0.40, 0.42, 0.44],
#     ('NEE', 'XGBoost'): [0.34, 0.37, 0.38],
#     ('NEE', 'RF'): [0.33, 0.35, 0.37],
#     ('GPP', 'Physics-aware transformer'): [0.77, 0.77, 0.78],
#     ('GPP', 'XGBoost'): [0.75, 0.75, 0.76],
#     ('GPP', 'RF'): [0.75, 0.74, 0.75],
#     ('RECO', 'Physics-aware transformer'): [0.74, 0.75, 0.75],
#     ('RECO', 'XGBoost'): [0.73, 0.74, 0.74],
#     ('RECO', 'RF'): [0.72, 0.74, 0.73]
# }
#
# # bias_data = {
# #     ('NEE', 'CFR'): [-0.05, -0.02, 0.02],
# #     ('NEE', 'XGBoost'): [0.00, 0.03, 0.08],
# #     ('NEE', 'RF'): [-0.04, -0.01, 0.04],
# #     ('GPP', 'CFR'): [0.03, 0.01, -0.05],
# #     ('GPP', 'XGBoost'): [-0.05, -0.06, -0.19],
# #     ('GPP', 'RF'): [-0.10, -0.10, -0.24],
# #     ('RECO', 'CFR'): [-0.01, -0.01, -0.02],
# #     ('RECO', 'XGBoost'): [-0.06, -0.03, -0.12],
# #     ('RECO', 'RF'): [-0.13, -0.10, -0.18]
# # }
# # 绝对值
# bias_data = {
#     ('NEE', 'Physics-aware transformer'): [0.05, 0.02, 0.02],
#     ('NEE', 'XGBoost'): [0.00, 0.03, 0.08],
#     ('NEE', 'RF'): [0.04, 0.01, 0.04],
#     ('GPP', 'Physics-aware transformer'): [0.03, 0.01, 0.05],
#     ('GPP', 'XGBoost'): [0.05, 0.06, 0.19],
#     ('GPP', 'RF'): [0.10, 0.10, 0.24],
#     ('RECO', 'Physics-aware transformer'): [0.01, 0.01, 0.02],
#     ('RECO', 'XGBoost'): [0.06, 0.03, 0.12],
#     ('RECO', 'RF'): [0.13, 0.10, 0.18]
# }

def bar_multi(metrics_data, variables, models, Y_metrics):
    plt.rcParams['font.family'] = 'Arial'
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    bar_width = 0.3
    outputs = ['Output1', 'Output2', 'Output3']

    # Modify the label for the 'Physics-aware transformer '
    modified_models = ['Physics-aware\ntransformer' if model == 'Physics-aware transformer' else model for model in
                       models]

    for i, metric_data in enumerate(metrics_data):
        ax = axes[i]
        for k, variable in enumerate(variables):
            for j, model in enumerate(modified_models):
                x = k * len(models) + j
                values = metric_data[(variable, models[j])]
                for m, output in enumerate(outputs):
                    label_offset = 0.001 if values[m] >= 0 else -0.015
                    # Define lighter colors, as mentioned in your function
                    # lighter_greens = ['#262626', '#293890', '#BF1D2D']
                    # lighter_greens = ['black', '#589FF3', '#F94141']
                    # colors1 = ['black', 'blue', 'red']  # Define colors for plots
                    # colors2 = ['black', '#589FF3', '#F94141']  # Lighter colors
                    # lighter_greens = []
                    # for c1, c2 in zip(colors1, colors2):
                    #     rgb1 = mcolors.to_rgb(c1)
                    #     rgb2 = mcolors.to_rgb(c2)
                    #     new_rgb = [(r1 + r2) / 2 for r1, r2 in zip(rgb1, rgb2)]
                    #     lighter_greens.append(mcolors.to_hex(new_rgb))
                    # lighter_greens = ['#262626', '#23B2E0', '#EF232B']
                    # lighter_greens = ['#7A7A7A', '#5EB0D2', '#D93A3B']
                    # 中#7A7A7A 是一种中灰色。#5EB0D2 是一个较深的蓝绿色。 #D93A3B 是一个适中的红色
                    lighter_greens = ['#A0A0A0', '#8EC4D4', '#F36B6F']
                    # 浅
                    # lighter_greens = ['black', 'blue','red']
                    ax.bar(x + m * bar_width, values[m], bar_width, label=output, color=lighter_greens[m])
                    # Format the bar value to two decimal places
                    ax.text(x + m * bar_width, values[m] + label_offset, f'{values[m]:.2f}', ha='center', va='bottom',fontsize=12)
                if k < len(variables) - 1:
                    ax.axvline((k + 1) * len(models) - 0.2, color='black', linestyle='--', linewidth=1)
            # Add variable labels
            if i == 0:
                ax.text(k * len(models) + 1, 1.73, variable, fontsize=16, ha='center', va='center') # with meteorological
                # ax.text(k * len(models) + 1, 1.85, variable, fontsize=16, ha='center',
                #         va='center')  # without meteorological
        ax.tick_params(axis='y', labelsize=13)  # Sets the font size for both x and y axis tick labels
        ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
        ax.set_xticklabels([model for model in modified_models] * len(variables), fontsize=15)
        ax.set_ylabel(Y_metrics[i], fontsize=16, fontname='Arial')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        # Set y-limits based on the row
        if i == 0:  # 第一行
            # ax.set_ylim(1.00, max(max(values) for values in metric_data.values())+0.05)  # Correct this line
            ax.set_ylim(1.00, 1.70)  # Correct this line
        elif i == 1:  # 第二行
            # ax.set_ylim(0.45, max(max(values) for values in metric_data.values())+0.05)  # with meteorological data
            ax.set_ylim(0.25, 0.85)  # with meteorological data
            # ax.set_ylim(0.40, max(max(values) for values in metric_data.values()) + 0.05)  # without meteorological data
        elif i == 2:  # 第三行
        # ax.set_ylim(0, max(max(values) for values in metric_data.values()) + 0.05)  # 示例设置
        #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置小数点格式为两位
            ax.set_ylim(0.00, 0.21)

        # handles, labels = ax.get_legend_handles_labels()
        # unique_labels = list(dict.fromkeys(labels))  # Remove duplicate labels
        # unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
        # ax.legend(handles=unique_handles, labels=unique_labels, loc='lower right')

    plt.tight_layout()
    plt.show()

    results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(results_path + 'Fig 0404' + '.png', dpi=300)

# Prepare data
variables = ['NEE', 'GPP', 'RECO']
models = ['Physics-aware transformer', 'XGBoost', 'RF']
Y_metrics = ['RMSE', 'CC', 'Absolute Bias']

# RMSE, CC, and Bias data
metrics_data = [rmse_data, cc_data, bias_data]

# Call function with data and plot
bar_multi(metrics_data, variables, models, Y_metrics)

# def bar_multi(metrics_data, variables, models,Y_metrics):
#     plt.rcParams['font.family'] = 'Arial'
#     fig, axes = plt.subplots(3, 1, figsize=(12, 14))
#     bar_width = 0.3
#     outputs = ['Output1', 'Output2', 'Output3']
#
#     for i, metric_data in enumerate(metrics_data):
#         ax = axes[i]
#         for k, variable in enumerate(variables):
#             for j, model in enumerate(models):
#                 x = k * len(models) + j
#                 values = metric_data[(variable, model)]
#                 # for m, output in enumerate(outputs):
#                 #     ax.bar(x + m * bar_width, values[m], bar_width, label=output,
#                 #            color=plt.cm.viridis(m / len(outputs)))
#                 #     ax.text(x + m * bar_width, values[m] + 0.005, str(values[m]), ha='center', va='bottom')
#                 for m, output in enumerate(outputs):
#                     label_offset = 0.002 if values[m] >= 0 else -0.015  # Adjust y-coordinate based on the sign of the value
#                     # # ax.bar(x + m * bar_width, values[m], bar_width, label=output,color=plt.cm.viridis(m / len(outputs)))
#                     # light_cmap = plt.cm.get_cmap('viridis', 100)  # Get a lighter colormap
#                     # colors = light_cmap(m / len(outputs))  # Obtain a lighter color from the colormap
#                     # # Replace the previous ax.bar line with the following code
#                     # ax.bar(x + m * bar_width, values[m], bar_width, label=output, color=colors)
#                     # Define lighter green colors
#                     # lighter_greens = ['#2c7fb8','#41b6c4','#a1dab4',]  # Example lighter green colors
#                     lighter_greens = ['#262626', '#BF1D2D', '#293890', ]  # Example lighter BLACK colors
#                     # Replace the previous ax.bar line with the following code
#                     ax.bar(x + m * bar_width, values[m], bar_width, label=output, color=lighter_greens[m])
#                     ax.text(x + m * bar_width, values[m] + label_offset, str(values[m]), ha='center', va='bottom')
#             if k < len(variables) - 1:
#                    ax.axvline((k + 1) * len(models) - 0.2, color='black', linestyle='--',linewidth=1)
#             # 添加变量标签
#             if i == 0:
#                 ax.text(k * len(models) + 1, 1.8, variable, fontsize=12,ha='center', va='center')
#         ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
#         ax.set_xticklabels([f'{model}' for model in models] * len(variables), fontsize=12)
#         # ax.set_xlabel('Models', fontname='Arial')
#         ax.set_ylabel(Y_metrics[i], fontsize=12,fontname='Arial')
#         # ax.set_title(f'{list(metric_data.values())[0]} by {list(metric_data.keys())[0]}')
#         handles, labels = ax.get_legend_handles_labels()
#         unique_labels = list(dict.fromkeys(labels))  # Remove duplicate labels
#         unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
#         ax.legend(handles=unique_handles, labels=unique_labels, loc='lower right')
#     # fig.text(0.5, 0.01, 'Models', ha='center', va='center', fontsize=12,
#     #          fontname='Arial')
#     plt.tight_layout()
#     plt.show()
#     resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
#     plt.savefig(resluts_path + 'Absolute three outputs ' + 'using RLFM v3' + '.png', dpi=300)
#
# # Prepare data
# variables = ['NEE', 'GPP', 'RECO']
# models = ['Physics-aware transformer ', 'XGBoost', 'RF']
# Y_metrics = ['RMSE','CC','Absolute Bias']
#
# # RMSE, CC, and Bias data
# metrics_data = [rmse_data, cc_data, bias_data]
#
# # Call function with data and plot
# bar_multi(metrics_data, variables, models,Y_metrics)
# resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
# plt.savefig(resluts_path + 'Absolute three outputs ' + 'using RLFM v3' + '.png', dpi=300)


# def bar_multi(rmse_data,variables,models):
#     for i, variable in enumerate(variables):
#         for j, model in enumerate(models):
#             x = i * len(models) + j
#             values = rmse_data[(variable, model)]
#             for k, output in enumerate(outputs):
#                 ax.bar(x + k * bar_width, values[k], bar_width, label=output, color=plt.cm.viridis(k / len(outputs)))
#                 # 添加数值标签
#                 ax.text(x + k * bar_width, values[k] + 0.03, str(values[k]), ha='center', va='bottom')
#
#         # 添加变量标签
#         ax.text(i * len(models) + 1, 1.8, variable, fontsize=12,
#                 ha='center', va='center')
#
#         if i < len(variables) - 1:
#             ax.axvline((i + 1) * len(models) - 0.25, color='black', linestyle='--')
#
#     ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
#     ax.set_xticklabels([f'{model}' for model in models] * len(variables))
#     ax.set_xlabel('Models', fontname='Arial')
#     ax.set_ylabel('RMSE', fontname='Arial')
#
#     # # 设置图例
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = list(dict.fromkeys(labels))  # 去除重复的标签
#     unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
#     ax.legend(handles=unique_handles, labels=unique_labels, loc='upper right', bbox_to_anchor=(1, 1))
#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 准备数据
# variables = ['NEE', 'GPP', 'RECO']
# models = ['CFR', 'XGBoost', 'RF']
# outputs = ['Output1', 'Output2', 'Output3']
#
# # RMSE, CC, and Bias data
# rmse_data = {
#     ('NEE', 'CFR'): [1.10, 1.13, 1.11],
#     ('NEE', 'XGBoost'): [1.14, 1.17, 1.15],
#     ('NEE', 'RF'): [1.14, 1.16, 1.13],
#     ('GPP', 'CFR'): [1.53, 1.56, 1.53],
#     ('GPP', 'XGBoost'): [1.58, 1.66, 1.62],
#     ('GPP', 'RF'): [1.57, 1.65, 1.63],
#     ('RECO', 'CFR'): [1.30, 1.29, 1.27],
#     ('RECO', 'XGBoost'): [1.34, 1.35, 1.32],
#     ('RECO', 'RF'): [1.34, 1.33, 1.31]
# }
#
# cc_data = {
#     ('NEE', 'CFR'): [0.53, 0.54, 0.54],
#     ('NEE', 'XGBoost'): [0.47, 0.48, 0.48],
#     ('NEE', 'RF'): [0.48, 0.49, 0.50],
#     ('GPP', 'CFR'): [0.81, 0.81, 0.81],
#     ('GPP', 'XGBoost'): [0.79, 0.77, 0.78],
#     ('GPP', 'RF'): [0.79, 0.78, 0.77],
#     ('RECO', 'CFR'): [0.79, 0.80, 0.79],
#     ('RECO', 'XGBoost'): [0.77, 0.77, 0.76],
#     ('RECO', 'RF'): [0.77, 0.77, 0.77]
# }
#
# bias_data = {
#     ('NEE', 'CFR'): [-0.04, -0.01, 0.04],
#     ('NEE', 'XGBoost'): [-0.03, 0.00, 0.05],
#     ('NEE', 'RF'): [-0.04, 0.00, 0.06],
#     ('GPP', 'CFR'): [0.01, 0.01, -0.06],
#     ('GPP', 'XGBoost'): [-0.02, -0.03, -0.17],
#     ('GPP', 'RF'): [-0.06, -0.06, -0.20],
#     ('RECO', 'CFR'): [-0.02, 0.01, -0.02],
#     ('RECO', 'XGBoost'): [-0.05, 0.00, -0.07],
#     ('RECO', 'RF'): [-0.10, -0.06, -0.13]
# }
#
# # 设置图形参数
# bar_width = 0.25
# x_positions = np.arange(len(variables) * len(models))
# metrics_data = [rmse_data, cc_data, bias_data]
# # 绘制柱形图
# fig, ax = plt.subplots(figsize=(12, 8))
# for i in range(len(variables)):
#     bar_multi(metrics_data[i],variables,models)
# plt.tight_layout()
# plt.show()
# for i, variable in enumerate(variables):
#     for j, model in enumerate(models):
#         x = i * len(models) + j
#         values = rmse_data[(variable, model)]
#         for k, output in enumerate(outputs):
#             ax.bar(x + k * bar_width, values[k], bar_width, label=output, color=plt.cm.viridis(k / len(outputs)))
#             # 添加数值标签
#             ax.text(x + k * bar_width, values[k] + 0.03, str(values[k]), ha='center', va='bottom')
#
#     # 添加变量标签
#     ax.text(i * len(models) + 1, 1.8, variable,fontsize=12,
#             ha='center', va='center')
#
#     if i < len(variables) - 1:
#            ax.axvline((i + 1) * len(models) - 0.25, color='black', linestyle='--')
#
# ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
# ax.set_xticklabels([f'{model}' for model in models] * len(variables))
# ax.set_xlabel('Models', fontname='Arial')
# ax.set_ylabel('RMSE', fontname='Arial')
#
# # # 设置图例
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = list(dict.fromkeys(labels))  # 去除重复的标签
# unique_handles = [h[0] for h in zip(handles, labels) if h[1] in unique_labels]
# ax.legend(handles=unique_handles, labels=unique_labels, loc='upper right', bbox_to_anchor=(1, 1))

# plt.tight_layout()
# plt.show()

# # 准备数据
# variables = ['NEE', 'GPP', 'RECO']
# models = ['CFR', 'XGBoost', 'RF']
# outputs = ['Output1', 'Output2', 'Output3']
#
# rmse_data = {
#     ('NEE', 'CFR'): [1.10, 1.13, 1.11],
#     ('NEE', 'XGBoost'): [1.14, 1.17, 1.15],
#     ('NEE', 'RF'): [1.14, 1.16, 1.13],
#     ('GPP', 'CFR'): [1.53, 1.56, 1.53],
#     ('GPP', 'XGBoost'): [1.58, 1.66, 1.62],
#     ('GPP', 'RF'): [1.57, 1.65, 1.63],
#     ('RECO', 'CFR'): [1.30, 1.29, 1.27],
#     ('RECO', 'XGBoost'): [1.34, 1.35, 1.32],
#     ('RECO', 'RF'): [1.34, 1.33, 1.31]
# }
#
# # 设置图形参数
# bar_width = 0.25
# bar_spacing = 0.1
# x_positions = np.arange(len(variables) * len(models))
# plt.rcParams['font.family'] = 'Arial'
# # 绘制纵向柱状图
# fig, ax = plt.subplots(figsize=(12, 8))
#
# for i, variable in enumerate(variables):
#     for j, model in enumerate(models):
#         x = i * len(models) + j
#         values = rmse_data[(variable, model)]
#         for k, output in enumerate(outputs):
#             ax.bar(x + k * bar_width, values[k], bar_width, label=output, color=plt.cm.viridis(k / len(outputs)))
#             # 添加数值标签
#             ax.text(x + k * bar_width, values[k] + 0.03, str(values[k]), ha='center', va='bottom')
#
#     # 添加变量标签
#     ax.text(i * len(models) + 1, 1.8, variable,fontsize=12,
#             ha='center', va='center')
#
#     # if i < len(variables) - 1:
#     #     ax.axvline((i + 1) * len(models) - 0.25, color='black', linestyle='--')
#
# # 设置坐标轴标签和图例
# ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
# ax.set_xticklabels([f'{model}' for model in models] * len(variables))
# ax.set_xlabel('Models',fontsize=12)
# ax.set_ylabel('RMSE',fontsize=12,)
# plt.legend(outputs, loc='upper right', bbox_to_anchor=(1, 1))
# plt.tight_layout()
#
# plt.show()

# # 准备数据
# variables = ['NEE', 'GPP', 'RECO']
# models = ['CFR', 'XGBoost', 'RF']
# outputs = ['Output1', 'Output2', 'Output3']
#
# rmse_data = {
#     ('NEE', 'CFR'): [1.10, 1.13, 1.11],
#     ('NEE', 'XGBoost'): [1.14, 1.17, 1.15],
#     ('NEE', 'RF'): [1.14, 1.16, 1.13],
#     ('GPP', 'CFR'): [1.53, 1.56, 1.53],
#     ('GPP', 'XGBoost'): [1.58, 1.66, 1.62],
#     ('GPP', 'RF'): [1.57, 1.65, 1.63],
#     ('RECO', 'CFR'): [1.30, 1.29, 1.27],
#     ('RECO', 'XGBoost'): [1.34, 1.35, 1.32],
#     ('RECO', 'RF'): [1.34, 1.33, 1.31]
# }
#
# # 设置图形参数
# bar_width = 0.25
# bar_spacing = 0.1
# x_positions = np.arange(len(variables) * len(models))
#
# # 绘制纵向柱状图
# fig, ax = plt.subplots(figsize=(12, 8))
#
# for i, variable in enumerate(variables):
#     for j, model in enumerate(models):
#         x = i * len(models) + j
#         values = rmse_data[(variable, model)]
#         for k, output in enumerate(outputs):
#             ax.bar(x + k * bar_width, values[k], bar_width, label=output, color=plt.cm.viridis(k / len(outputs)))
#             # 添加数值标签
#             ax.text(x + k * bar_width, values[k] + 0.03, str(values[k]), ha='center', va='bottom')
#
#     # 添加变量标签
#     ax.text(i * len(models) + 1, max([rmse_data[(variable, model)][-1] for model in models]) + 0.3, variable,
#             ha='center', va='center')
#
#     if i < len(variables) - 1:
#         ax.axvline((i + 1) * len(models) - 0.25, color='black', linestyle='--')
#
# # 设置坐标轴标签和图例
# ax.set_xticks(np.arange(len(variables) * len(models)) + 0.3)
# ax.set_xticklabels([f'{model}' for model in models] * len(variables))
# ax.set_xlabel('Models')
# ax.set_ylabel('RMSE')
# plt.legend(outputs, loc='upper right', bbox_to_anchor=(1.05, 1))
# plt.tight_layout()
#
# plt.show()
# # # 准备数据
# models = ['CFR', 'XGBoost', 'RF']
# outputs = ['Output1', 'Output2', 'Output3']
#
# # 统计值数据（这里以 RMSE 为例）
# rmse_data = {
#     'NEE': {
#         'CFR': [1.10, 1.13, 1.11],
#         'XGBoost': [1.14, 1.17, 1.15],
#         'RF': [1.14, 1.16, 1.13]
#     },
#     'GPP': {
#         'CFR': [1.53, 1.56, 1.53],
#         'XGBoost': [1.58, 1.66, 1.62],
#         'RF': [1.57, 1.65, 1.63]
#     },
#     'RECO': {
#         'CFR': [1.30, 1.29, 1.27],
#         'XGBoost': [1.34, 1.35, 1.32],
#         'RF': [1.34, 1.33, 1.31]
#     }
# }
#
# # 创建水平并列柱状图并标注数值
# fig, axs = plt.subplots(3, 1, figsize=(8, 10))
#
# for i, var in enumerate(rmse_data):
#     x = np.arange(len(models))
#     width = 0.2
#     for j, output in enumerate(outputs):
#         offsets = np.linspace(-width, width, len(models))
#         values = [rmse_data[var][model][j] for model in models]
#         for k, model in enumerate(models):
#             bar = axs[i].bar(x[k] + offsets[k] + j*0.07, values[k], width/len(outputs), label=output, align='center')
#             for b in bar:
#                 height = b.get_height()
#                 axs[i].text(b.get_x() + b.get_width() / 2, height, round(height, 2), ha='center', va='bottom')
#
#     axs[i].set_title(f'Statistics for {var}')
#     axs[i].set_ylabel('Values')
#     axs[i].set_xticks(x)
#     axs[i].set_xticklabels(models)
#     axs[i].legend(title='Outputs', loc='upper right')
#     axs[i].set_xlabel('Models')
#
# plt.tight_layout()
# plt.show()

# # 统计变量数据
# data = {
#     'NEE_CFR': [1.10, 1.13, 1.11],
#     'NEE_XGBoost': [1.14, 1.17, 1.15],
#     'NEE_RF': [1.14, 1.16, 1.13],
#     'GPP_CFR': [1.53, 1.56, 1.53],
#     'GPP_XGBoost': [1.58, 1.66, 1.62],
#     'GPP_RF': [1.57, 1.65, 1.63],
#     'RECO_CFR': [1.30, 1.29, 1.27],
#     'RECO_XGBoost': [1.34, 1.35, 1.32],
#     'RECO_RF': [1.34, 1.33, 1.31]
# }
#
# # 创建整体图表
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# # 绘制 RMSE 图表
# axs[0].boxplot(data.values())
# axs[0].set_xticklabels(data.keys(), rotation=45)  # 旋转标签
# axs[0].set_title('RMSE')
#
# # 绘制 CC 图表
# axs[1].boxplot(data.values())
# axs[1].set_xticklabels(data.keys(), rotation=45)  # 旋转标签
# axs[1].set_title('CC')
#
# # 绘制 Bias 图表
# axs[2].boxplot(data.values())
# axs[2].set_xticklabels(data.keys(), rotation=45)  # 旋转标签
# axs[2].set_title('Bias')
#
# plt.show()