
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd

# ====================== 第一部分：栅格数据对齐预处理 ======================
def align_raster_to_reference(reference_path, input_path, output_path, resample_method=Resampling.bilinear):
    """
    将输入栅格对齐到参考栅格的空间属性（坐标系、范围、分辨率）
    :param reference_path: 参考栅格路径（土地覆盖图）
    :param input_path: 待对齐的输入栅格路径（如 NEE、GPP 等）
    :param output_path: 对齐后的输出栅格路径
    :param resample_method: 重采样方法（默认双线性插值）
    """
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile  # 获取参考栅格的元数据
        ref_crs = ref_profile['crs']
        ref_transform = ref_profile['transform']
        ref_width = ref_profile['width']
        ref_height = ref_profile['height']
        ref_bounds = ref.bounds

    with rasterio.open(input_path) as src:
        # 获取输入栅格的数据类型和nodata值
        src_dtype = src.dtypes[0]  # 保留输入栅格的原始数据类型
        src_nodata = src.nodata     # 保留输入栅格的原始nodata值

        # 创建目标栅格数据容器（使用输入栅格的数据类型）
        dst_data = np.zeros((ref_profile['count'], ref_height, ref_width), dtype=src_dtype)

        # 执行重投影和裁剪
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_resolution=(ref_transform.a, abs(ref_transform.e)),
            resampling=resample_method
        )

        # 更新元数据并保存对齐后的栅格
        ref_profile.update({
            'dtype': src_dtype,     # 使用输入栅格的数据类型
            'nodata': src_nodata     # 使用输入栅格的nodata值
        })
        with rasterio.open(output_path, 'w', **ref_profile) as dst:
            dst.write(dst_data)

# ====================== 第二部分：分区统计核心功能 ======================
# def zonal_stats(land_cover_path, var_paths, stats_type='mean'):
#     with rasterio.open(land_cover_path) as lc_src:
#         lc = lc_src.read(1)
#         lc_nodata = lc_src.nodata
#
#     results = {}
#     classes = np.unique(lc[lc != lc_nodata])
#
#     for var_name, var_path in var_paths.items():
#         with rasterio.open(var_path) as var_src:
#             var_data = var_src.read(1)
#             var_nodata = var_src.nodata
#
#         stats = {}
#         for cls in classes:
#             # 修复1：保持二维掩膜结构
#             mask_2d = (lc == cls)
#
#             # 修复2：直接在原始数据上应用双重过滤
#             valid_mask = (var_data != var_nodata) & mask_2d
#
#             # 修复3：直接提取有效值
#             valid_values = var_data[valid_mask]
#
#             if stats_type == 'mean':
#                 stats[cls] = np.nanmean(valid_values) if valid_values.size > 0 else np.nan
#             elif stats_type == 'sum':
#                 stats[cls] = np.nansum(valid_values) if valid_values.size > 0 else np.nan
#             else:
#                 raise ValueError("stats_type 必须为 'mean' 或 'sum'")
#
#         results[var_name] = stats
#
#     df = pd.DataFrame(results)
#     df.index.name = 'LandCoverClass'
#     return df

# 先对NEE取绝对值再取平均值
def zonal_stats(land_cover_path, var_paths, stats_type='mean'):
    with rasterio.open(land_cover_path) as lc_src:
        lc = lc_src.read(1)
        lc_nodata = lc_src.nodata

    results = {}
    classes = np.unique(lc[lc != lc_nodata])  # 只选取有效的土地覆盖类别

    for var_name, var_path in var_paths.items():
        with rasterio.open(var_path) as var_src:
            var_data = var_src.read(1)
            var_nodata = var_src.nodata

        stats = {}
        for cls in classes:
            # 生成掩膜
            mask_2d = (lc == cls)
            valid_mask = (var_data != var_nodata) & mask_2d
            valid_values = var_data[valid_mask]

            if valid_values.size > 0:
                # NEE 变量取绝对值后再计算
                if "NEE" in var_name:
                    valid_values = np.abs(valid_values)

                if stats_type == 'mean':
                    stats[cls] = np.nanmean(valid_values)
                elif stats_type == 'sum':
                    stats[cls] = np.nansum(valid_values)
                else:
                    raise ValueError("stats_type 必须为 'mean' 或 'sum'")
            else:
                stats[cls] = np.nan  # 没有数据时，返回 NaN

        results[var_name] = stats

    df = pd.DataFrame(results)
    df.index.name = 'LandCoverClass'
    return df


# 调用示例
if __name__ == '__main__':
    # 定义路径
    land_cover_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/LC_Types/Exclude_lc_without_data/LC_without_13_15_17.tif"
    raw_data_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Flux come/Extract_by_LC/'
    aligned_data_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/aligned_data/'
    # raw_data_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/Extract_by_LC/'
    # aligned_data_dir = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/aligned_data/'

    # 创建对齐后的数据目录
    os.makedirs(aligned_data_dir, exist_ok=True)

    # 需要处理的变量列表
    variables = ['NEE_ERA5_year_by_LC', 'GPP_ERA5_year_by_LC', 'TER_ERA5_year_by_LC','Abs_NEE-(RECO-GPP)_ERA5_year_by_LC']
    # variables = ['DL_NEE_year_by_LC', 'DL_GPP_year_by_LC', 'DL_RECO_year_by_LC', 'DL_abs_NEE-(RECO-GPP)_year_by_LC']

    # Step 1: 将所有变量栅格对齐到土地覆盖图
    for var in variables:
        input_path = os.path.join(raw_data_dir, f'{var}.tif')
        output_path = os.path.join(aligned_data_dir, f'{var}_aligned.tif')
        align_raster_to_reference(land_cover_path, input_path, output_path)

    # Step 2: 执行分区统计
    var_paths = {
        var: os.path.join(aligned_data_dir, f'{var}_aligned.tif')
        for var in variables
    }

    # 统计平均值
    df_mean = zonal_stats(land_cover_path, var_paths, stats_type='mean')
    print("平均值统计结果:")
    print(df_mean)
    # df_mean.to_csv('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/Zonal_statistics_by_LC/DL_abs_zonal_stats_mean_v2.csv')

    # 统计总和
    df_sum = zonal_stats(land_cover_path, var_paths, stats_type='sum')
    print("\n总和统计结果:")
    print(df_sum)
    # df_sum.to_csv('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/Zonal_statistics_by_LC/DL_abs_zonal_stats_sum_v2.csv')

    # # 保存为 CSV
    df_mean.to_csv('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/Zonal_statistics_by_LC/FLUXCOM_zonal_stats_mean_v2.csv')
    df_sum.to_csv('E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/Zonal_statistics_by_LC/FLUXCOM_zonal_stats_sum_v2.csv')