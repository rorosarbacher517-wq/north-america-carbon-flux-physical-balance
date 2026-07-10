import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv("E:/The North America ecosystem carbon flux/Sites_estimation/data/Sites_csv/Five folds 257.csv")

# 统计 'Vegetation' 字段每一类别的个数
vegetation_counts = df['Vegetation'].value_counts()

# 输出结果
print(vegetation_counts)

plt.rcParams['font.family'] = 'Arial'
# # 可视化为饼图
# plt.figure(figsize=(8, 6))
# plt.pie(vegetation_counts, labels=vegetation_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Distribution of Vegetation Categories', fontsize=16)
# plt.axis('equal')  # 保证饼图是圆形
# plt.show()
# 可视化为柱状图
# 可视化为柱状图
plt.figure(figsize=(6, 4))
plt.bar(vegetation_counts.index, vegetation_counts.values, color='skyblue')
plt.title('The counts of each plant functional type', fontsize=20)  # 加粗标题
plt.tick_params(axis='both', labelsize=16)  # 加粗刻度标签
plt.xlabel('Plant functional types (PFTs)', fontsize=20)  # 加粗 x 轴标签
plt.ylabel('Count', fontsize=20)  # 加粗 y 轴标签
plt.xticks(rotation=45)  # 加粗 x 轴标签
plt.grid(axis='y')  # 添加 y 轴网格
plt.tight_layout()  # 调整布局

# 保存图像
resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/sites_distribution/'
plt.savefig(resluts_path + 'The counts of each plant functional type2' + '.png', dpi=400)

# 显示图形
plt.show()

# ENF    61
# GRA    43
# DBF    42
# CRO    40
# WET    25
# OSH    19
# MF      6
# SAV     6
# CSH     6
# WSA     4
# BSV     2
# CVM     2
# SNO     1



