
import glob
import pandas as pd


import pandas as pd
import os


# 定义处理 Excel 文件的函数
def process_excel(file_path):
    data = pd.read_excel(file_path)  # 读取 Excel 文件
    data['观看次数'] = data['您观看的报告场次（必填）'].apply(lambda x: len(x.split(',')))
    return data


# 指定文件夹路径
folder_path = "E:/Personal infomation/个人重要文件/博士阶段资料/24年暑期学校签到表-樊彬/单个文件/"
all_data = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        processed_data = process_excel(file_path)
        all_data.append(processed_data[['您的真实姓名（课程期间，请保证每次填写一致）（必填）','您所在的单位（课程期间，请保证每次填写一致）（必填）','观看次数']])


# 根据姓名 单位统计’签到次数‘；根据姓名，单位分组，并对观看次数求和，得到每个人总的观看次数，最后将统计结果放到一个csv里
# 将所有表拼接成一个
combined_data = pd.concat(all_data, ignore_index=True)
# 重命名列
combined_data.rename(columns={'您的真实姓名（课程期间，请保证每次填写一致）（必填）': '姓名', '您所在的单位（课程期间，请保证每次填写一致）（必填）': '单位'}, inplace=True)

# 按照姓名和单位进行分组，并统计同一姓名-单位出现的数量
result_combined_data = combined_data.groupby(['姓名', '单位']).size().reset_index(name='签到次数')

# # 将result_count 和 LC 统计结果合并
# result_combined_data = pd.concat([result_count, result_count])

# 按照姓名和单位进行分组，并统计总的观看次数
result_sum = result_combined_data.groupby(['姓名', '单位']).agg({'签到次数':'sum'}).reset_index()
result_sum.columns = ['姓名', '单位', '签到次数']  # 重命名列名

result_sum.to_csv(folder_path+'最终统计结果.csv', index=False, encoding='utf-8-sig')
