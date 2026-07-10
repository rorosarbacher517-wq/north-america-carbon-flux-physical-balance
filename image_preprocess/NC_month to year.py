
import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

# 指定 NetCDF 文件所在的文件夹
folder_path = r"E:\The North America ecosystem carbon flux\PPT_GRAPHS\Graphs\Mapping comparison\Flux come"
output_folder = os.path.join(folder_path, "Month_to_year")

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 每月对应的天数（非闰年）
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# 获取所有 .nc 文件
nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

# 遍历每个 .nc 文件进行处理
for file_name in nc_files:
    file_path = os.path.join(folder_path, file_name)

    try:
        # 读取 NetCDF 文件
        ds = xr.open_dataset(file_path, engine="netcdf4")

        # 选择主要变量（假设是第一个数据变量）
        var_name = list(ds.data_vars.keys())[0]
        da = ds[var_name]  # 形状应为 (time, lat, lon)

        # 检查是否为12个月
        if da.sizes["time"] != 12:
            raise ValueError(f"{file_name} 中的时间维度不是12个月。")

        # 将每个月的值乘以对应的天数后求和（单位转换为 g C m⁻² yr⁻¹）
        weighted_sum = sum(da.isel(time=i) * days_in_month[i] for i in range(12))

        # 保留属性
        weighted_sum.attrs = da.attrs

        # 获取空间坐标信息（假设是 lat 和 lon）
        lat = ds["lat"].values
        lon = ds["lon"].values
        transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

        # 输出文件名
        output_tif = os.path.join(output_folder, f"{file_name[:-3]}_annual.tif")

        # 写入 GeoTIFF 文件
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=weighted_sum.shape[0],
            width=weighted_sum.shape[1],
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(weighted_sum.values.astype(np.float32), 1)

        print(f"✔ 处理完成: {file_name} -> {output_tif}")

    except Exception as e:
        print(f"✘ 处理失败: {file_name}, 错误信息: {e}")
