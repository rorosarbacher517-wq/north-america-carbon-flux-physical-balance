
import os
import glob

import pandas as pd
import pyproj
import rasterio
from osgeo import gdal
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from datetime import datetime
import h5py
from rasterio.errors import RasterioIOError

# *************************************
# 去填充值、cirrus，云、云阴影和雪
# Create a function to convert each element to binary and determine the mask value
def convert_to_binary(qa):
    binary = bin(qa)[2:]  # 将qa转换为二进制
    mask = False
    # 检查指定位置的位是否为1
    if len(binary) > 5 and any(binary[-bit] == '1' for bit in [0, 2, 3, 4, 5]):
        mask = True
    return mask, binary

def binary_mask_8(qa):
    # 原来bit0，2, 3, 4, 5 分别对应于原始数据的填充值，cirrus，cloud，cloudshadow，snow，
    # 转换为二进制之后有16个有效位，
    qa_int = int(qa) # 将输入值qa转换为整数类型
    binary_qa = np.binary_repr(qa_int) #将整数值 qa_int 转换为二进制字符串
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    bit2 = (binary_qa >> 2) & 1
    bit3 = (binary_qa >> 3) & 1
    bit4 = (binary_qa >> 4) & 1
    bit5 = (binary_qa >> 5) & 1
    return bit0 == 1 or bit2 ==1 or bit3 == 1 or bit4 == 1 or bit5 == 1  # 如果bit0、bit3、bit4或bit5中至少有一个为1，则返回True。如果以上所有位的值都为0，则返回False。

def binary_mask_457(qa):
    # 原来bit0，3,4,5分别对应于填充值，cloud，cloudshadow，snow，457没有cirrus
    # 由于bit2,14,15 这三个位unused，转换为二进制之后只有13个有效位
    # 因而填充值对应于bit0不变，cloud，cloudshadow，snow向前移一位，应对应于bit2,3,4
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    bit2 = (binary_qa >> 2) & 1
    bit3 = (binary_qa >> 3) & 1
    bit4 = (binary_qa >> 4) & 1
    return bit0 == 1 or bit2 == 1 or bit3 == 1 or bit4 == 1

def read_qaband_tif(QA_PIXEL_path):
    QA_PIXEL_path = QA_PIXEL_path[0]
    # 将\\换为/
    QA_PIXEL_path = eval(repr(QA_PIXEL_path).replace('\\', '/'))
    QA_PIXEL_path = QA_PIXEL_path.replace('//', '/')
    QA_PIXEL_dataset = gdal.Open(QA_PIXEL_path, gdal.GA_ReadOnly)
    # 读取QA_PIXEL影像数据
    qa_band = QA_PIXEL_dataset.GetRasterBand(1)
    qa_data = qa_band.ReadAsArray()
    # Convert qa_data to integer type
    qa_data_int = qa_data.astype(int)
    return qa_data,qa_data_int

# 根据经纬度和地理信息返回行列号
def lon_lat_to_row_col(lon, lat, geo_information):
    a = np.array([[geo_information[1], geo_information[2]], [geo_information[4], geo_information[5]]])
    b = np.array([lon - geo_information[0], lat - geo_information[3]])
    r = np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
    # 上面注释掉的col和row是有小数点的，需要用整数，所以用int()向下取整。
    col = int(r[0])
    row = int(r[1])
    return col, row

def resubset_values(point,img_tif):
    print('hhh')
    # 计算point在tif文件中的行列号位置坐标
    # 创建投影坐标系的转换器
    proj_in = pyproj.CRS.from_epsg(4326)  # 输入坐标系为WGS84经纬度坐标系
    proj_out = pyproj.CRS.from_epsg(3857)  # 输出坐标系为EPSG:3857投影坐标系
    # # 将经纬度坐标转换为投影坐标
    # 定义转换器
    transformer = pyproj.Transformer.from_crs(proj_in, proj_out, always_xy=True)
    # 进行坐标转换
    x, y = transformer.transform(point.x, point.y)
    # 在此处编写代码以获取 single_band_data, im_proj, im_geotrans
    single_band_data = None  # 假设默认值为 None
    im_proj = None
    im_geotrans = None
    # 打开影像文件
    with rasterio.open(img_tif) as src:
        print(src.crs)
        # 判断 `img_tif` 是否为 `None`
        if src.crs is None:
            print('src is None')
        else:
            # 将点地理坐标转换为像素坐标
            point_coords = src.index(x, y)
            # 获取行列号（像素坐标）
            row, col = int(point_coords[0]), int(point_coords[1])
            print(row,col)
            print(src.height, src.width)

            # 定义影像块的大小
            block_size = 3  # 影像块的大小为50x50

            # 计算影像块的左上角像素坐标和宽高
            left = col - block_size // 2
            top = row - block_size // 2
            width = block_size
            height = block_size

            # 从原始影像中读取指定大小的影像块
            window = Window(left, top, width, height)
            single_band_data = src.read(1, window=window)  # 读取第一个波段的像素值
            print(single_band_data.shape)
            # 在这里判断window_data是不是空的，如果是需要跳过进行处理
            # 获取投影信息和仿射变换信息
            im_proj = src.crs.to_string()  # 获取投影信息
            # im_geotrans = src.transform.to_gdal()  # 获取的是原影像的仿射变换信息
            # 获取从窗口提取的影像块data的仿射变换信息
            window_transform = src.window_transform(window)

            # 将windows提取的影像块的仿射变换信息转换为GDAL格式
            im_geotrans = window_transform.to_gdal()

    return single_band_data,im_proj,im_geotrans

# def resubset_values(img_tif):
#     # 打开影像文件
#     with rasterio.open(img_tif) as src:
#         # 计算影像img_tif的中心像素行列号
#         row = src.height // 2
#         col = src.width // 2
#
#         # 定义小影像块的大小为3x3，以中心像素为中心
#         block_size = 3
#         left = col - block_size // 2
#         top = row - block_size // 2
#         width = block_size
#         height = block_size
#
#         # 限制小影像块的范围在合法范围内
#         left = max(0, left)
#         top = max(0, top)
#         right = min(src.width, left + width)
#         bottom = min(src.height, top + height)
#
#         # 获取指定大小的小影像块
#         window = Window(left, top, right - left, bottom - top)
#         single_band_data = src.read(1, window=window)  # 读取第一个波段的像素值
#
#         # 获取投影信息和仿射变换信息
#         im_proj = src.crs.to_string()  # 获取投影信息
#         im_geotrans = src.window_transform(window).to_gdal()  # 获取小影像块的仿射变换信息
#
#     return single_band_data, im_proj, im_geotrans

def read_multiband(imagepath,point,bandlist,mask):
    # 将 mask 转换为布尔类型
    mask_cloud = mask.astype(bool)
    # 查找匹配的文件路径
    files = []
    for keyword in bandlist:
        files += glob.glob(f'{imagepath}/*{keyword}*.tif')
    # 打开多波段影像
    re_datasets = []
    for file in files:
        # 将dataset提取为50*50大小
        dataset,im_proj,im_geotrans = resubset_values(point, file)
        # dataset, im_proj, im_geotrans = resubset_values(file)
        # 将dataset中为填充值，云 云阴影 雪的值转为-9999，其余正常值返回为影像像素值
        # 如果 mask 中的元素为真（非零），则对应位置的 masked_dataset 的值将被设置为 -9999，否则保持不变。
        # masked_dataset = np.where(mask, -9999, dataset)
        # 创建一个副本 masked_dataset，初始值为 dataset
        masked_dataset = np.copy(dataset)
        print(file)
        # 使用布尔掩码将 dataset 中对应位置的像素值设置为 -9999
        masked_dataset[mask_cloud] = -9999
        # 将不等于-9999的转换为地表反射率
        reflect_dataset = np.where(masked_dataset != -9999, (masked_dataset * 0.0000275) - 0.2, masked_dataset)
        re_datasets.append(reflect_dataset)
    # list转为ndarray
    re_datasets = np.array(re_datasets)
    return re_datasets,im_proj,im_geotrans

def write_multiband(re_datasets,site,year,doy,im_proj,im_geotrans,data_year_path):
    im_bands,im_width,im_height = re_datasets.shape
    # 将影像数据输出为tif
    data_name = site + '_' + str(year) + '_' + str(doy) + '.tif'
    data_path = os.path.join(data_year_path, data_name)
    # 生成tif遥感影像
    driver = gdal.GetDriverByName('GTiff')  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(data_path, im_width, im_height, im_bands, gdal.GDT_Float32)
    if dataset is None:
        print("the dataset is None.")
    else:
        dataset.SetProjection(im_proj)  # 写入投影
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(re_datasets)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(re_datasets[i])
        del dataset

def wrire_image_hdf(x_array,site,year,data_year_path):
    print('hhh')
    x_shape = x_array.shape
    x_lenth = x_array.shape[0]
    x_data_name = site + '_' + str(year)+ '_' +str(x_lenth) +'.h5'
    x_data_path = os.path.join(data_year_path, x_data_name)
    # 创建HDF5文件
    file = h5py.File(x_data_path, 'w')
    image_data = file.create_dataset('image_data', shape=x_shape, dtype=np.float32)
    # 将数据写入数据波段
    image_data[:,:,:,:] = x_array

# 将日期转为数字
def out_day_by_date(Date):
    '''
    根据输入的日期计算该日期是在当年的第几天
    '''
    print(Date)
    # 将字符串转为日期格式
    Date = datetime.strptime(Date, '%Y_%m_%d').date()
    # date = Date.apply(lambda x: datetime.strptime(x,'%Y%m%d'))
    print(Date)
    year = Date.year
    month = Date.month
    day = Date.day
    # year = date[0:4]
    # month = date[4:6]
    # day = int(date[6:8])
    months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if 0 < month <= 12:
        sum = months[month - 1]
    else:
        print("month error")
    sum += day
    leap = 0
    # 接下来判断平年闰年
    if (year % 400 == 0) or ((year % 4) == 0) and (year % 100 != 0):  # and的优先级大于or
        # 1、世纪闰年:能被400整除的为世纪闰年
        # 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
        leap = 1
    if (leap == 1) and (month > 2):
        sum += 1  # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    return year,sum,Date

# 将每个站点&每一年的数据拼接起来组成一个输入数据
def concatenate_time_series(list_of_arrays):
    # 检查列表是否为空
    if len(list_of_arrays) == 0:
        print("x_list is empty")
        combined_array = -9999
    else:
        max_shape = max(arr.shape[0] for arr in list_of_arrays)  # 找到最大的shape的第0个维度
        new_list = []
        for arr in list_of_arrays:
            if arr.shape[0] < max_shape:
                padding_shape = (max_shape - arr.shape[0],) + arr.shape[1:]
                padding = np.full(padding_shape, -9999)
                new_arr = np.concatenate((arr, padding), axis=0)
                new_list.append(new_arr)
            else:
                new_list.append(arr)
        combined_array = np.array(new_list)
    return combined_array

def read_tif(point,file_path):
    QA_PIXEL_path = eval(repr(file_path).replace('\\', '/'))
    QA_PIXEL_path = QA_PIXEL_path.replace('//', '/')
    dataset,im_proj,im_geotrans = resubset_values(point, QA_PIXEL_path)
    return dataset,im_proj,im_geotrans

def FparExtra_QC_binary_mask(qa):
    # FparExtra_QC中 2, 4, 5,6 分别对应于原始数据的填充值，snow,cirrus，cloud，cloudshadow，
    # 转换为二进制之后有7个有效位，
    qa_int = int(qa) # 将输入值qa转换为整数类型
    binary_qa = np.binary_repr(qa_int) #将整数值 qa_int 转换为二进制字符串
    binary_qa = int(binary_qa)
    bit2 = (binary_qa >> 2) & 1
    bit4 = (binary_qa >> 4) & 1
    bit5 = (binary_qa >> 5) & 1
    bit6 = (binary_qa >> 6) & 1
    return bit2 ==1 or bit4 == 1 or bit5 == 1 or bit6 == 1  # 如果bit0、bit3、bit4或bit5中至少有一个为1，则返回True。如果以上所有位的值都为0，则返回False。

def remove_fpar_lai_clouds(point, Fpar_file_path, Lai_file_path,FparLai_QC_file_path):
    # 将\\换为/
    Fpar_data,Fpar_proj,Fpar_geotrans = read_tif(point,Fpar_file_path)
    Lai_data,Lai_proj,Lai_geotrans = read_tif(point,Lai_file_path)
    FparExtra_QC_data, FparLai_QC_proj, FparLai_QC_geotrans = read_tif(point, FparLai_QC_file_path)
    # 将qa_data中为1的位置视为云像素位mask，然后将对应位置的re_data数据标记为-9999
    print(FparExtra_QC_data.shape,Fpar_data.shape,Lai_data)
    # 标记值bit 456 为1或值为-9999的位置：Cirrus,Internal_CloudMask,Cloud_Shadow，-9999
    # 将QC值转为二进制
    FparExtra_QC_data_int = FparExtra_QC_data.astype(int)
    mask_FparExtra_QC = np.zeros_like(FparExtra_QC_data, dtype=bool)
    mask_FparExtra_QC = np.vectorize(FparExtra_QC_binary_mask, otypes=[int])(FparExtra_QC_data_int)
    print(mask_FparExtra_QC)

    cloud_mask = (FparLai_QC_data == 1) | (qa_data == -9999)
    # 将对应位置的 re_data 修改为 -9999
    re_data[cloud_mask] = -9999
    print(re_data)
    # 将不等于-9999的转换为地表反射率
    re_dataset = np.where(re_data != -9999, re_data * 0.0001, re_data)

    return re_dataset,re_proj,re_geotrans


if __name__ == '__main__':
    # 下面的功能主要是对fluxnet变量进行聚合，从半小时聚合到天和月，然后是提取一定窗口大小的影像50*50
    # 最终输出要输入模型的x_image_array和y_flux_array
    shp_path = "E:/Carbon_flux/data/originate_data/site_shp/Sites_HH_GE_shar_6.shp"
    # 读取shp文件
    points = gpd.read_file(shp_path)
    # 总的data array，后面每添加一个x_array就添加一个time_array和site_array
    x_list = []
    y_list = []
    # 利用窗口重采样之后 影像数据存储的位置
    re_image_root_path = 'E:/Carbon_flux/data/sites_image3_resample'
    # 聚合后的Y变量存储的路径
    Y_df_root_path = 'E:/Carbon_flux/data/originate_data/GE_AmeriFlux_data/GE_AmeriFlux_data_DD_MM'
    # site_image存储根路径
    site_image_path = 'E:/Carbon_flux/data/sites_image3/lai'
    # 进入site目录
    sites_list = os.listdir(site_image_path)
    # 进入
    for i in range(0, 2):
        print('hhh')
        site = sites_list[i]
        point = points.geometry[i]
        lon = point.x
        lat = point.y
        site_year_path = os.path.join(site_image_path, sites_list[i])
        os.chdir(site_year_path)
        # 进入site_year list
        site_year_list = os.listdir(site_year_path)
        for j in range(4,len(site_year_list)):
            imagelistpath = os.path.join(site_year_path, site_year_list[j])
            print('hhh')
            os.chdir(imagelistpath)
            # 进入year iamge目录
            # 创建站点文件夹
            data_site_name = site
            data_site_path = os.path.join(re_image_root_path, data_site_name)
            # 检查文件夹是否存在
            if not os.path.isdir(data_site_path):
                # 创建文件夹
                os.makedirs(data_site_path)
                print(f"已创建文件夹：{data_site_path}")
            else:
                print(f"文件夹已存在：{data_site_path}")
            data_year_name = site_year_list[j]
            data_year_path = os.path.join(data_site_path, data_year_name)
            # 检查文件夹是否存在
            if not os.path.isdir(data_year_path):
                # 创建文件夹
                os.makedirs(data_year_path)
                print(f"已创建文件夹：{data_year_path}")
            else:
                print(f"文件夹已存在：{data_year_path}")
            x_year_list = []  # 每一年的array
            time_list = []  # 每一年数据对应的年 doy 日期
            site_list = []  # 每一年数据对应的siteID longitude latitude
            year_imag_list = os.listdir(imagelistpath)
            year_imag_list = list(set(year_imag_list))
            Fpar_doy_list = []
            Lai_doy_list = []
            for k in range(0,len(year_imag_list)):
                imagepath = os.path.join(imagelistpath, year_imag_list[k])
                os.chdir(imagepath)
                # 进入特定天多波段影像存储的目录
                print('hhh')
                # 依次遍历bandlist，取对应的反射率波段和QA波段
                bandlist = ['Fpar', 'Lai', 'FparLai_QC', 'FparExtra_QC','FparStdDev','LaiStdDev']
                re_doy_list = []
                # 读取该路径下的qa波段,返回qa_mask
                Fpar_filename = f'*Fpar.tif'
                Lai_filename = f'*Lai.tif'
                FparLai_QC_filename = f'*FparExtra_QC.tif'

                Fpar_file_list = glob.glob(os.path.join(imagepath, Fpar_filename))
                Lai_file_list = glob.glob(os.path.join(imagepath, Lai_filename))
                FparLai_QC_file_list = glob.glob(os.path.join(imagepath, FparLai_QC_filename))

                Fpar_file_path = Fpar_file_list[0]
                Lai_file_path = Lai_file_list[0]
                FparLai_QC_file_path = FparLai_QC_file_list[0]


                # 读取反射率波段和QA波段数据，进行去云
                Fpar_datasets, Lai_datasets, im_proj, im_geotrans = remove_fpar_lai_clouds(point, Fpar_file_path, Lai_file_path,FparLai_QC_file_path)
                Fpar_doy_list.append(Fpar_datasets)
                Lai_doy_list.append(Lai_datasets)


                for i in range(1,len(bandlist)+1):  # 假设文件序号从1到7
                    reflectance_filename = f'*Reflectance_Band{i}*'
                    quality_filename = f'*Quality_Band{i}*'
                    reflectance_file_list = glob.glob(os.path.join(imagepath, reflectance_filename))
                    quality_file_list = glob.glob(os.path.join(imagepath, quality_filename))
                    reflectance_file_path = reflectance_file_list[0]
                    quality_file_path = quality_file_list[0]
                    # 读取反射率波段和QA波段数据，进行去云
                    re_datasets, im_proj, im_geotrans = remove_clouds(point,reflectance_file_path,quality_file_path)
                    re_doy_list.append(re_datasets)  # 将一天的7个波段的影像添加到一个list
                x_year_list.append(re_doy_list)# 将这一年的每一天的7个波段的影像添加到一个list
                # 返回影像的年份和doy
                year, doy, Date = out_day_by_date(year_imag_list[k][0:10])
                time_list.append((year, doy))
                # 返回影像对应的site序列号、经度和维度
                site_list.append((site, lon, lat))
                # 将获得的有效影像输出 存储为tif 一天多个波段 命名为site_year_doy
                # 将re_doy_list转为array
                re_doy_mulbandarray = np.array(re_doy_list)
                write_multiband(re_doy_mulbandarray, site, year, doy, im_proj, im_geotrans, data_year_path)
            # 每一年的单独记录命名为site_year_doy_t 存储为tif  (t 与doy不一样 t是从1到n的记录索引数 小于等于一年内有效影像的最大值)
            print(x_year_list)
            # max_shape = max(a.shape for a in x_year_list)
            # x_year_list_adjusted = [np.pad(a,[(0, max_shape[0] - a.shape[0]), (0, max_shape[1] - a.shape[1]), (0, max_shape[2] - a.shape[2])],
            #            mode='constant', constant_values=-9999) for a in x_year_list]
            # 将列表中的数组拼接为一个ndarray
            x_array = np.stack(x_year_list)
            # x_array = np.stack(x_year_list)
            x_list.append(x_array)
            # print(x_array[1, :, 1, 1])
            wrire_image_hdf(x_array, site, year, data_year_path)

            site_array = np.asarray(site_list)
            # print(site_array[1, :])

            time_array = np.asarray(time_list)
            # print(time_array[1, :])

            time_df = pd.DataFrame(time_array, columns=['Year', 'DoY'])
            site_df = pd.DataFrame(site_array[:, :], columns=['Site_Id','longitude','latitude'])
            # 实际的站点-影像-doy
            Sites_time_df = pd.concat([site_df, time_df], axis=1)
            # Y_df_path_name = site + '_DD_MM_' + str(year) + '.csv'
            # # 遍历指定路径及其子文件夹下的所有文件
            # for root, dirs, files in os.walk(Y_df_root_path):
            #     for file in files:
            #         # 检查文件名是否匹配
            #         if file == Y_df_path_name:
            #             # 构建完整文件路径
            #             file_path = os.path.join(root, file)
            #
            #             # 打开文件并读取数据
            #             Y_df = pd.read_csv(file_path)
            #             print(Y_df.head())
            #
            #             # 将实际的X_df(Sites_time_df)与Y_df 根据'Site_Id', 'Year', 'DoY' 进行匹配
            #             Flux_image_Y_df = pd.merge(Sites_time_df, Y_df, on=['Site_Id', 'Year', 'DoY'], how='left')
            #             print(len(Flux_image_Y_df))
            #             # 将没有匹配值的设置为-9999
            #             Flux_image_Y_df = Flux_image_Y_df.fillna(-9999)
            #             print(Flux_image_Y_df.head())
            #             Flux_image_Y_df_name = 'Y_' + site + '_' + str(year) + '.csv'
            #             # Flux_image_Y_df_root_path = 'E:/2_biomass/data/test_data/flux/flux_DD_MM/Y_Data'
            #             Flux_image_Y_df_root_path = 'E:/Carbon_flux/data/originate_data/GE_AmeriFlux_data/y_data'
            #             Flux_image_Y_df_path = os.path.join(Flux_image_Y_df_root_path, Flux_image_Y_df_name)
            #             Flux_image_Y_df.to_csv(Flux_image_Y_df_path)
            #             y_list.append(np.array(Flux_image_Y_df))
    # print('hhh')
    x_image_array = concatenate_time_series(x_list)
    max_shape = max(a.shape for a in x_list)
    x_year_list_adjusted = [
        np.pad(a, [(0, max_shape[0] - a.shape[0]), (0, max_shape[1] - a.shape[1]), (0, max_shape[2] - a.shape[2])],
               mode='constant', constant_values=-9999) for a in x_list]
    x_year_array = np.stack(x_year_list_adjusted)
    print(x_image_array.shape)
    y_image_array = concatenate_time_series(y_list)
    # print(y_image_array.shape)
    inputdata_path = 'E:/Carbon_flux/data/input_data/'
    np.save(inputdata_path + 'x_image_array_modis.npy', x_image_array)
    # np.save(inputdata_path + 'y_image_array2.npy', y_image_array)