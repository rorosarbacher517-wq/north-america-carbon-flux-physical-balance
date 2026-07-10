import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_standardized_residuals(raster1, raster2, output_path):
    """计算两个栅格文件的标准残差（Standardized Residuals），并保存为新的栅格文件"""

    # 读取栅格数据
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        data1 = src1.read(1).astype(float)  # 读取第一层数据，并转换为 float 类型
        data2 = src2.read(1).astype(float)

        # 处理无效值（NoData 处理）
        nodata_value = src1.nodata if src1.nodata is not None else -9999
        data1[data1 == nodata_value] = np.nan
        data2[data2 == nodata_value] = np.nan

        # 计算标准差（避免标准差过小）
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)  # 仅计算有效像元
        all_pixels = np.concatenate((data1[valid_mask], data2[valid_mask]))  # 提取所有有效数据
        std_dev = np.std(all_pixels) + 1e-6  # 避免除零

        # 计算标准残差
        z = np.full_like(data1, np.nan)  # 先创建 NaN 矩阵
        z[valid_mask] = (data1[valid_mask] - data2[valid_mask]) / std_dev  # 仅计算有效值

        # 设定输出栅格的参数
        profile = src1.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)  # 确保输出也是 float 类型，并保留 NoData

        # 保存为新栅格
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(z.astype(np.float32), 1)

    print("标准残差栅格已保存:", output_path)
    return output_path


from rasterio.warp import reproject, Resampling


def align_rasters(target_raster_path, base_raster_path, output_path):
    """将目标栅格对齐到基准栅格的网格"""
    with rasterio.open(base_raster_path) as base:
        base_profile = base.profile.copy()
        base_data = base.read(1)

        with rasterio.open(target_raster_path) as target:
            # 创建目标数组
            target_aligned = np.empty_like(base_data)

            # 重采样
            reproject(
                source=target.read(1),
                destination=target_aligned,
                src_transform=target.transform,
                src_crs=target.crs,
                dst_transform=base.transform,
                dst_crs=base.crs,
                resampling=Resampling.bilinear
            )

            # 保存对齐后的栅格
            with rasterio.open(output_path, 'w', **base_profile) as dst:
                dst.write(target_aligned, 1)
    return output_path




def visualize_raster(raster_path, title="Standardized Residuals Map"):
    """读取并可视化栅格数据"""

    with rasterio.open(raster_path) as src:
        data = src.read(1)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='RdBu', vmin=-3, vmax=3)  # 颜色方案适合正负对比
    plt.colorbar(label="Standardized Residuals (Z-score)")
    plt.title(title)
    plt.axis("off")  # 关闭坐标轴
    plt.show()


def calculate_nrmse(raster1_path, raster2_path, output_path):
    """
    计算归一化均方根误差（NRMSE），并保存结果为新的栅格文件。

    参数：
    raster1_path : str : 第一个栅格文件路径（观测值）
    raster2_path : str : 第二个栅格文件路径（预测值）
    output_path : str : 输出NRMSE栅格文件路径
    """

    # 读取栅格数据
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        data1 = src1.read(1).astype(float)  # 读取第一层数据，并转换为 float 类型
        data2 = src2.read(1).astype(float)

        # 处理无效值（NoData 处理）
        nodata_value1 = src1.nodata if src1.nodata is not None else -9999
        nodata_value2 = src2.nodata if src2.nodata is not None else -9999
        data1[data1 == nodata_value1] = np.nan
        data2[data2 == nodata_value2] = np.nan

        # 创建有效掩码（只计算有效像元）
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)

        # 计算均方根误差（RMSE）
        mse = np.nanmean((data1[valid_mask] - data2[valid_mask]) ** 2)  # 计算有效样本的均方差
        rmse = np.sqrt(mse)  # 计算RMSE

        # 计算观测值的均值（也只在有效像元内）
        mean_observed = np.nanmean(data1[valid_mask])  # 计算观测值的均值

        # 计算NRMSE
        if mean_observed == 0:
            nrmse = np.nan  # 避免除以零
        else:
            nrmse = rmse / mean_observed

        # 创建输出NRMSE栅格
        output_data = np.full(data1.shape, np.nan)  # 初始化输出数组为NaN
        output_data[valid_mask] = nrmse  # 仅在有效值位置填入计算结果

        # 设定输出栅格的参数
        profile = src1.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)  # 确保输出也是 float 类型，并保留 NoData

        # 保存NRMSE结果为新的栅格文件
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_data.astype(np.float32), 1)

    print("NRMSE栅格已保存:", output_path)


def compute_nrmse_per_pixel(raster1_path, raster2_path, output_path, norm_method="range"):
    """
    逐像元计算归一化均方根误差（NRMSE）

    参数:
        raster1_path : str : 第一个栅格文件路径
        raster2_path : str : 第二个栅格文件路径
        output_path : str : 输出NRMSE栅格文件路径
        norm_method : str : 归一化方法，可选 "range", "mean", "std", "max_abs"
    """

    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        A = src1.read(1).astype(float)
        B = src2.read(1).astype(float)

        # 处理无效值
        A[A == src1.nodata] = np.nan
        B[B == src2.nodata] = np.nan

        # 创建有效掩码（只计算有效像元）
        valid_mask = ~np.isnan(A) & ~np.isnan(B)
        # 计算有效像元的总数
        num_valid_pixels = np.sum(valid_mask)
        print(f"有效像元的数量: {num_valid_pixels}")

        # 逐像元计算RMSE和标准差
        output_data = np.full(A.shape, np.nan)  # 初始化输出数组为NaN

        # 循环逐像元计算
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                if valid_mask[row, col]:
                    # 取出对应像元的值
                    a_val = A[row, col]
                    b_val = B[row, col]

                    # RMSE计算
                    rmse = np.sqrt((a_val - b_val) ** 2)  # 对应像元的RMSE
                    # 对象值（基于有效值计算的归一化因子）
                    if norm_method == "range":
                        norm_factor = np.nanmax([A[valid_mask], B[valid_mask]]) - np.nanmin(
                            [A[valid_mask], B[valid_mask]])
                    elif norm_method == "mean":
                        norm_factor = np.nanmean([A[valid_mask], B[valid_mask]])
                    elif norm_method == "std":
                        norm_factor = np.nanstd([A[valid_mask], B[valid_mask]])
                    elif norm_method == "max_abs":
                        norm_factor = max(np.nanmax([A[valid_mask], B[valid_mask]]),
                                          abs(np.nanmin([A[valid_mask], B[valid_mask]])))
                    else:
                        raise ValueError("不支持的归一化方法！可选：range, mean, std, max_abs")

                    # 避免除以零
                    norm_factor = norm_factor if norm_factor != 0 else 1e-6

                    # 存储标准差（NRMSE）
                    output_data[row, col] = rmse / norm_factor

        # 设定输出栅格的参数
        profile = src1.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)  # 确保输出也是 float 类型，并保留 NoData

        # 保存NRMSE结果为新的栅格文件
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_data.astype(np.float32), 1)

    print(f"逐像元NRMSE计算已完成，并保存至: {output_path}")


def read_raster_data(raster_path):
    """读取栅格数据并返回有效值"""
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        # 替换无效值为NaN
        data[data == src.nodata] = np.nan
    return data


def compute_correlation_and_plot(raster1_path, raster2_path):
    """计算两个栅格的相关系数并绘制散点密度图"""

    # 读取栅格数据
    data1 = read_raster_data(raster1_path)
    data2 = read_raster_data(raster2_path)

    # 创建有效掩码
    valid_mask = ~np.isnan(data1) & ~np.isnan(data2)

    # 选择有效像元
    valid_data1 = data1[valid_mask]
    valid_data2 = data2[valid_mask]

    # 计算相关系数
    correlation = np.corrcoef(valid_data1, valid_data2)[0, 1]
    print(f"相关系数: {correlation}")

    # 创建散点密度图
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=valid_data1, y=valid_data2, fill=True, cmap='Blues', thresh=0, levels=30)
    plt.scatter(valid_data1, valid_data2, color='red', alpha=0.5, s=10)  # 绘制点
    plt.title('散点密度图')
    plt.xlabel('Raster 1 Values')
    plt.ylabel('Raster 2 Values')
    plt.axis('equal')  # 使X和Y轴的比例相同
    plt.grid(True)
    plt.show()


# ==== 示例运行 ====
# 替换为你的实际栅格路径
raster1_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Physics_DL/resample/DL_NEE_year_by_LC_resample.tif"  # 替换为你的 GPP/RECO/NEE 数据
raster2_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Flux come/Extract by LC_daily/NEE_ERA5_year_by_LC.tif"
output_raster_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/DL-Fluxcom/standardized_residuals.tif"

aligned_raster1_path = "E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/Mapping comparison/Alaign/nrmse_range.tif"

align_rasters(raster1_path, raster2_path, aligned_raster1_path)
# compute_standardized_residuals(raster2_path, aligned_raster1_path, output_raster_path)
# 计算标准残差
# compute_standardized_residuals(raster1_path, raster2_path, output_raster_path)
compute_nrmse_per_pixel(aligned_raster1_path, raster2_path, output_raster_path, norm_method="max_abs")
compute_correlation_and_plot(aligned_raster1_path, raster2_path)
# 示例调用
# calculate_nrmse(aligned_raster1_path,raster2_path,output_raster_path)
# 可视化标准残差
# visualize_raster(output_raster_path)