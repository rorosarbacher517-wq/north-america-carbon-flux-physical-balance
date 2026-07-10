import os
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_origin

# 指定 NetCDF 文件所在的文件夹
folder_path = "E:/Paper code/data/Flux com/"  # 修改为你的文件夹路径
output_folder = os.path.join(folder_path, "output_tif")  # 存放转换后的 TIF 文件

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .nc 文件
nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

# 遍历每个 .nc 文件进行处理
for file_name in nc_files:
    file_path = os.path.join(folder_path, file_name)

    try:
        # 读取 NetCDF 文件
        ds = xr.open_dataset(file_path, engine="netcdf4")

        # 选择主要变量（假设是第一个数据变量）
        var_name = list(ds.data_vars.keys())[0]  # 获取数据变量名称
        da = ds[var_name]  # 提取数据变量

        # 计算 time 维度的平均值
        da_mean = da.mean(dim="time", keep_attrs=True)

        # 获取空间坐标信息（假设是 lat 和 lon）
        lat = ds["lat"].values
        lon = ds["lon"].values
        transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

        # 生成 TIF 文件名
        output_tif = os.path.join(output_folder, f"{file_name[:-3]}_mean.tif")

        # 保存为 GeoTIFF
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=da_mean.shape[0],
            width=da_mean.shape[1],
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",  # 假设使用 WGS84 坐标系
            transform=transform,
        ) as dst:
            dst.write(da_mean.values, 1)

        print(f" 处理完成: {file_name} -> {output_tif}")

    except Exception as e:
        print(f"处理失败: {file_name}, 错误信息: {e}")