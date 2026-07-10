import pandas as pd

# 读取CSV文件（请替换为你的文件路径）
df = pd.read_csv("E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/站点差异.csv")

# 提取两列的站点并转换为集合
set1 = set(df["site_list1"].dropna().astype(str))  # 处理空值并确保为字符串
set2 = set(df["site_list2"].dropna().astype(str))

# 计算差集：在set1中但不在set2中的元素
unique_to_list1 = list(set2 - set1)

# 打印结果
print(f"共找到 {len(unique_to_list1)} 个独特站点:")
print("\n".join(unique_to_list1))

# 可选：将结果保存到新CSV
pd.DataFrame(unique_to_list1, columns=["unique_sites"]).to_csv("E:/The North America ecosystem carbon flux/PPT_GRAPHS/Table/unique_sites_1.csv", index=False)
print("\n结果已保存到 unique_sites.csv")