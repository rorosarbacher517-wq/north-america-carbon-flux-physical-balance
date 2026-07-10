# 可视化天 月 年时间尺度的缺失值，折线图
# 分别读取 daily_missing_data_statistics.csv annual_missing_data_statistics.csv Monthly_missing_data_statistics.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14  # 默认字体大小

# 定义文件路径
paths = {
    "daily_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/daily_missing_data_statistics 0407.csv',
    "annual_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/annual_missing_data_statistics 0407.csv',
    "monthly_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/Monthly_missing_data_statistics 0407.csv',
    "daily_MODIS_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/daily_MODIS_missing_data_statistics 0407.csv',
    "annual_MODIS_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/annual_MODIS_missing_data_statistics 0407.csv',
    "monthly_MODIS_missing": 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/Monthly_MODIS_missing_data_statistics 0407.csv'
}

# 读取数据
dataframes = {key: pd.read_csv(path) for key, path in paths.items()}

# 计算缺失比例并乘以100%
for key in dataframes:
    dataframes[key]['Missing Ratio'] = dataframes[key]['Missing Ratio'] * 100.0

# 绘图函数
def plot_missing_ratio(ax, df, x_col, title, x_label):
    # 设置颜色为黑色并绘制
    sns.lineplot(data=df, x=x_col, y='Missing Ratio', color='black',label='Missing Percentage', ax=ax, marker='o', markeredgecolor='none')
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)  # 使用具体的x轴标签
    ax.set_ylabel('Missing Ratio (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)  # 设置刻度字体大小
    ax.legend(fontsize=14)
    ax.grid(linestyle='--', linewidth=0.7)  # 设置网格样式

# 设置绘图参数
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 第一行 - 碳通量缺失比例
plot_missing_ratio(axs[0, 0], dataframes['daily_missing'], 'DOY', 'Daily Carbon Flux Missing Percentage', 'DOY')
plot_missing_ratio(axs[0, 1], dataframes['monthly_missing'], 'Month', 'Monthly Carbon Flux Missing Percentage', 'Month')
plot_missing_ratio(axs[0, 2], dataframes['annual_missing'], 'Year', 'Annual Carbon Flux Missing Percentage', 'Year')

# 第二行 - MODIS 数据缺失比例
plot_missing_ratio(axs[1, 0], dataframes['daily_MODIS_missing'], 'DOY', 'Daily MODIS Missing Percentage', 'DOY')
plot_missing_ratio(axs[1, 1], dataframes['monthly_MODIS_missing'], 'Month', 'Monthly MODIS Missing Percentage', 'Month')
plot_missing_ratio(axs[1, 2], dataframes['annual_MODIS_missing'], 'Year', 'Annual MODIS Missing Percentage', 'Year')

# 调整布局
plt.tight_layout()
results_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Time series analysis/'
plt.savefig(results_path + 'Missing ratio of MODIS and Carbon flux 0407.png', dpi=300, bbox_inches='tight')
plt.show()