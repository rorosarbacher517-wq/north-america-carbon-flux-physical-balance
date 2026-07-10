import os
import pandas as pd

# image root path
# ref_image_root_path = 'E:/Carbon_flux/data/modis/sites_image_modis_ref_MCD43A4'
ref_image_root_path = 'E:/Carbon_flux/data/modis/Sites_image_ERA5_LAND_DAILY_AGGR'
# 进入站点list
empty_data_info = []

# 遍历站点目录
for site_dir in os.listdir(ref_image_root_path):
    site_path = os.path.join(ref_image_root_path, site_dir)

    # 遍历每个站点的年份目录
    for year_dir in os.listdir(site_path):
        year_path = os.path.join(site_path, year_dir)

        # 检查年份目录下是否有数据
        if len(os.listdir(year_path)) == 0:
            empty_data_info.append({
                'Site_Id': site_dir,
                'year': year_dir
            })

# 创建DataFrame并打印
df_empty_data = pd.DataFrame(empty_data_info)
print(df_empty_data.head())
print(df_empty_data)


# 与具有详细信息的df进行拼接
original_df = pd.read_csv("E:/Carbon_flux/data/input_data/Sites_HH_GE_shar_270.csv")

# Convert 'Site_Id' to the same data type in both dataframes
df_empty_data['Site_Id'] = df_empty_data['Site_Id']

# Now, you can attempt the merge again
df_nodownload = pd.merge(df_empty_data, original_df, on='Site_Id')

print(df_nodownload)
df_nodownload.to_csv('E:/Carbon_flux/data/modis/Sites_meteorology_nodownload.csv')
