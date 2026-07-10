# North America Carbon Flux Physical-Balance Code

这个目录整理自 `E:\The North America ecosystem carbon flux\Code`，用于保存北美生态系统碳通量估算中与遥感数据处理、气象数据准备、通量站点匹配、机器学习建模和碳通量物理平衡约束相关的代码。

整理原则：保留原始代码文件名和文件内容，只把适合进入 GitHub 的轻量源码、配置、说明和小型 CSV 表格复制进来。原始目录中的大体积中间数据、模型输入输出、缓存、IDE 状态和本地凭据没有纳入仓库。

## What this code does

从脚本结构看，这套代码围绕 NEE、GPP 和 RECO 三个碳通量变量展开。核心建模部分使用 TensorFlow/Keras 的 CNN/Transformer 类模型，并与 Random Forest、XGBoost 等基线模型比较。物理约束主要体现在自定义损失函数和验证脚本中，通过约束 `NEE + GPP - RECO` 的残差来比较带物理约束和不带物理约束的预测结果。

整体流程大致是：

1. 下载和裁剪 HLS、MODIS/VIIRS、Sentinel、ERA5-Land 等遥感和气象数据。
2. 对站点、影像、LAI/FPAR、反射率和气象变量进行时空匹配、插值、质量标记和块级预处理。
3. 构建站点级输入数组，训练深度学习、多输出机器学习和单变量基线模型。
4. 计算 NEE/GPP/RECO 预测精度、物理平衡残差、PFT 统计、时间序列验证和可视化结果。

## Directory map

| Directory | Purpose |
| --- | --- |
| `Carbon_flux/` | 碳通量建模、物理约束损失函数、训练脚本、站点/区域预测、模型比较和结果可视化。核心文件包括 `myloss.py`, `myloss_attention.py`, `Model_build.py`, `Model_build_attention.py`, `train_test.py`, `Physical constraints validation.py` 和多个 `Pro_*` 实验脚本。 |
| `ERA5_Land/` | ERA5-Land 数据下载和站点输入表准备，包含 CDS API 下载脚本。 |
| `GEE_Subset/` | Google Earth Engine 数据裁剪、MODIS/VIIRS/Sentinel/ERA5 子集下载、重采样、站点匹配和一个 `gee_subset_master` 工具包副本。 |
| `HLS/` | HLS 和 NASA AppEEARS 数据下载、HTTP 时间序列下载、质量控制、裁剪和站点批量下载脚本。 |
| `image_preprocess/` | 影像预处理、MODIS/LAI/FPAR 插值、气象匹配、分块、镶嵌、统计和像元/站点提取工具。 |

## Physical-balance constraint

物理约束主要在 `Carbon_flux/myloss.py` 和 `Carbon_flux/myloss_attention.py` 中实现。代码在常规预测误差之外加入 NEE、GPP、RECO 之间的平衡残差项，即约束预测值满足：

```text
NEE + GPP - RECO ~= 0
```

`Carbon_flux/Physical constraints validation.py` 进一步读取带物理约束和不带物理约束的模型输出，计算不同插值/质量控制条件下的 CFR residuals，用于比较物理约束对碳通量一致性的影响。

## Data and files not included

以下内容保留在原始工作目录中，没有复制到 GitHub 项目：

- `data/` 目录下的 `.npy` 模型输入、站点结果和预测输出。
- 大体积 GeoTIFF、中间影像、压缩包和运行产物。
- `__pycache__/` Python 缓存。
- `.idea/` 本地 IDE 工程文件。
- `HLS/.netrc`，该文件可能包含下载服务账号信息。

运行脚本前需要按原始代码中的相对路径准备数据，例如 `Carbon_flux` 下多处脚本默认读取 `./data/sites_input/`、`./data/sites_output/` 或 `./data/sites_results/`。

## Main dependencies

代码依赖按脚本导入大致包括：

- Core scientific stack: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`
- Deep learning: `tensorflow`, `keras`, `tensorflow_addons`
- Machine learning: `xgboost`, `shap`
- Geospatial/raster: `rasterio`, `rioxarray`, `xarray`, `geopandas`, `shapely`, `pyproj`, `gdal/osgeo`, `earthpy`
- Remote sensing services: `earthaccess`, `appeears`, `ee`, `geemap`, `cdsapi`
- Utilities: `requests`, `bs4`, `h5py`, `tqdm`, `psutil`, `scienceplots`
- Some scripts reference `arcpy`, so those scripts require an ArcGIS Python environment.

## Notes

- The source scripts were copied without content edits. Some comments in the original files appear to have encoding mojibake; they were intentionally left unchanged.
- Many scripts are experiment snapshots with hard-coded local paths or relative `data/` assumptions. Treat this directory as an organized code archive rather than a packaged Python library.
- Before publishing new derived outputs, keep large arrays, rasters, credentials and local environment files outside Git, or use a data repository/object storage workflow.
