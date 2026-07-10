import numpy as np
import os

import pandas as pd
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import griddata


def ref_binary_mask(qa):
    # 原来bit0，3,4,5分别对应于填充值，cloud，cloudshadow，snow，457没有cirrus
    # 由于bit2,14,15 这三个位unused，转换为二进制之后只有13个有效位
    # 因而填充值对应于bit0不变，cloud，cloudshadow，snow向前移一位，应对应于bit2,3,4
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    return bit0 == 0, bit0 == 1


# Assuming 'df_band is the name of your dataframe
def interpolate_row(row):
    f = interpolate.interp1d(range(len(row)), row, kind='cubic')
    return f(np.linspace(0, len(row) - 1, len(row)))


def ref_time_series_interpolation(array_data, band_index, fillvalue_bounds):
    if band_index in [0, 1, 2, 3, 4, 5, 6]:
        array_data = array_data.astype(float)
        array_data[array_data == -9999] = np.nan
        df_band = pd.DataFrame(array_data)
        df_band_linear = df_band.interpolate(method='linear', axis=1, limit_direction='both')
        # 处理0-1之外的除了-9999之外的值为空值然后使用nearest插值
        # Step 1: Select rows based on the condition
        condition = (df_band_linear < fillvalue_bounds[0]) | (df_band_linear > fillvalue_bounds[1])
        # Step 2: Replace the selected values with -9999
        df_band_linear[condition] = np.nan
        df_band_linear = df_band_linear.fillna(-9999)
        array_inter = np.array(df_band_linear)
    else:
        array_data = array_data
        # 将array_data中的-9999值用0替换
        array_data[array_data == -9999] = 0
        array_inter = array_data
    return array_inter


def ref_interpolation(ref_image_array):
    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = ref_image_array.copy()
    # check_imagemaxmin(interpolated_array)
    interpolated_array[:, :, 7:, :, :][interpolated_array[:, :, 7:, :, :] == 1] = -9999
    interpolated_array = full_filledvalue(interpolated_array)
    # 添加一个特征，用于表示待插值的和未插值的
    # 创建一个形状为1802, 366, 1, 3, 3的全零数组
    interpolated_array1 = np.zeros((interpolated_array.shape[0], interpolated_array.shape[1], 1, 3, 3))
    # 标识质量
    # 利用条件索引来修改new_array
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] != -9999] = 0
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] == -9999] = 1
    print(interpolated_array1.shape)
    interpolated_array1[:, :, 0, 1, 1]
    interpolated_array[:, :, 7, 1, 1]
    interpolated_array2 = interpolated_array.copy()
    # 对每个波段进行每一年的时间序列插值"
    for band_index in range(interpolated_array2.shape[2]):
        for ii in range(interpolated_array2.shape[3]):
            for jj in range(interpolated_array2.shape[4]):
                array_data = interpolated_array2[:, :, band_index, ii, jj]
                if band_index < 7:
                    fillvalue_bounds = (0, 1)
                else:
                    fillvalue_bounds = "extrapolate"
                interpolated_array2[:, :, band_index, ii, jj] = ref_time_series_interpolation(array_data,
                                                                                              band_index,
                                                                                              fillvalue_bounds)
    # 将空值用-9999来填充
    interpolated_array2[np.isnan(interpolated_array2)] = -9999
    interpolated_array2 = full_filledvalue(interpolated_array2)
    interpolated_array3 = np.concatenate((interpolated_array2, interpolated_array1), axis=2)
    return interpolated_array3


def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array


def ref_preprocess(x_ref_array):
    x_ref_array_mark = np.copy(x_ref_array)  # 复制输入数组，以保留原始数据
    # 将空值替换为-9999
    x_ref_array_mark[np.isnan(x_ref_array_mark)] = -9999
    # 将数组转换为 float64 类型
    x_ref_array_mark = x_ref_array_mark.astype('float64')
    #### 针对前七个地表反射率波段 将不等于-9999的乘以比例因子0.0001
    x_ref_array_mark[:, :, 0:7, :, :][x_ref_array_mark[:, :, 0:7, :, :] != -9999] *= 0.0001
    # 将ref_image_array所有波段里  反射率值不在0-3.2766范围内的像素值赋值为-9999。因为mcd43a4的最大\最小值即为32766\0,比例因子为0.0001, 将不在范围内的值赋值为-9999
    ref_mask1 = (x_ref_array_mark[:, :, :7, :, :] < 0) | (x_ref_array_mark[:, :, :7, :, :] > 1)
    ref_mask2 = (x_ref_array_mark[:, :, 7:, :, :] < 0)
    x_ref_array_mark[:, :, :7, :, :][ref_mask1] = -9999
    x_ref_array_mark[:, :, 7:, :, :][ref_mask1] = -9999
    x_ref_array_mark[:, :, :7, :, :][ref_mask2] = -9999
    x_ref_array_mark[:, :, 7:, :, :][ref_mask2] = -9999
    #### 进行质量标识
    ref_qa_array = x_ref_array_mark[:, :, 7:, :, :]
    ref_qa_array_mark = ref_binary_conversion_and_mark(ref_qa_array)
    # 将对应的mark array与影像合并
    ref_image_array = np.concatenate((x_ref_array_mark[:, :, :7, :, :], ref_qa_array_mark), axis=2)
    #### 对ref_array进行插值
    ref_image_array_inter = ref_interpolation(ref_image_array)
    ref_image_array_inter[:, :, 0, 1, 1]
    #### 检查是否有波段填充值数量不一致的情况 若有则将该像素所有的波段都视为填充值
    ref_image_array_full = full_filledvalue(ref_image_array_inter)
    return ref_image_array_full


def lai_binary_conversion_and_mark(lai_qa_array):
    # 将不等于-9999的每个值转为二进制
    # 将值转为整数
    # np.binary_repr函数得到的二进制字符串是按正常的方式从左到右表示的，即最高位（最左边）对应最高的权值，最低位（最右边）对应最低的权值。
    # 创建新的数组并进行标记
    FparLai_QC_mask = np.vectorize(FparLai_QC_binary_mask)(lai_qa_array[:, :, 0, :, :])
    # FparExtra_QC的值需要分别乘以比例因子 FparStdDev	0(min)	100(max)	0.01(scale)；LaiStdDev 0(min)	100(max)	0.1(scale)
    FparExtra_QC_mask = np.vectorize(FparExtra_QC_binary_mask)(lai_qa_array[:, :, 1, :, :])
    # print(FparLai_QC_mask,FparExtra_QC_mask)
    qa_lai_array_mark = np.zeros_like(lai_qa_array[:, :, 0, :, :])  # 创建一个和 qa_ref_array 相同形状的全0数组
    qa_lai_array_mark[lai_qa_array[:, :, 0, :, :] == -9999] = -9999  # 将对应的位置标记为-9999
    qa_lai_array_mark[
        (lai_qa_array[:, :, 0, :, :] != -9999) & (FparLai_QC_mask | FparExtra_QC_mask)] = 1  # 将视为坏数据的位置标记为1
    qa_lai_array_mark[
        (lai_qa_array[:, :, 0, :, :] != -9999) & ~FparLai_QC_mask & ~FparExtra_QC_mask] = 0  # 将视为好数据的位置标记为0
    # qa_lai_array_mark reshape 为[:,:,1,:,:]
    expanded_qa_lai_array_mark = np.reshape(qa_lai_array_mark,
                                            (qa_lai_array_mark.shape[0], qa_lai_array_mark.shape[1], 1,
                                             qa_lai_array_mark.shape[2], qa_lai_array_mark.shape[3]))
    # 检查改变后的形状
    return expanded_qa_lai_array_mark


def time_series_interpolation(array_data, band_index, fillvalue_bounds):
    if band_index in [0, 1]:
        array_data[array_data == 0] = -9999
        array_data = array_data.astype(float)
        array_data[array_data == -9999] = np.nan
        df_band = pd.DataFrame(array_data)
        df_band_linear = df_band.interpolate(method='linear', axis=1, limit_direction='both')
        df_band_linear
        # 处理0-1之外的除了-9999之外的值为空值然后使用nearest插值
        # Step 1: Select rows based on the condition
        condition = (df_band_linear <= fillvalue_bounds[0]) | (df_band_linear >= fillvalue_bounds[1])
        # Step 2: Replace the selected values with -9999
        df_band_linear[condition] = np.nan
        df_band_linear = df_band_linear.fillna(-9999)
        array_inter = np.array(df_band_linear)
    else:
        array_data = array_data
        # 将array_data中的-9999值用0替换
        array_data[array_data == -9999] = 0
        array_inter = array_data
    return array_inter


def lai_interpolation(fpar_lai_array):
    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = fpar_lai_array.copy()
    # 插值完之后 ，将mark标记仍为1的视为-9999
    interpolated_array[:, :, 2:, :, :][interpolated_array[:, :, 2:, :, :] == 1] = -9999
    interpolated_array = full_filledvalue(interpolated_array)
    # # 添加一个特征，用于表示待插值的和未插值的
    # # 创建一个形状为1802, 366, 1, 3, 3的全零数组
    interpolated_array1 = np.zeros((interpolated_array.shape[0], interpolated_array.shape[1], 1, 3, 3))
    # 标识质量
    # 利用条件索引来修改new_array
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] != -9999] = 0
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] == -9999] = 1
    print(interpolated_array1.shape)
    interpolated_array1[:, :, 0, 1, 1]
    interpolated_array[:, :, 2, 1, 1]
    interpolated_array2 = interpolated_array.copy()
    # 对每个波段进行每一年的时间序列插值"
    for band_index in range(interpolated_array2.shape[2]):
        for ii in range(interpolated_array2.shape[3]):
            for jj in range(interpolated_array2.shape[4]):
                array_data = interpolated_array2[:, :, band_index, ii, jj]
                if band_index == 0:
                    fillvalue_bounds = (0.0, 1.0)
                elif band_index == 1:
                    fillvalue_bounds = (0.0, 10.0)
                else:
                    fillvalue_bounds = "extrapolate"
                interpolated_array2[:, :, band_index, ii, jj] = time_series_interpolation(array_data, band_index,
                                                                                          fillvalue_bounds)
    # 将空值用-9999来填充
    interpolated_array2[np.isnan(interpolated_array2)] = -9999
    interpolated_array2 = full_filledvalue(interpolated_array2)
    interpolated_array3 = np.concatenate((interpolated_array2, interpolated_array1), axis=2)
    interpolated_array3 = full_filledvalue(interpolated_array3)
    return interpolated_array3


def lai_preprocess(x_lai_array):
    x_lai_array[:, :, 0, 1, 1]
    x_lai_array_mark = np.copy(x_lai_array)  # 复制输入数组，以保留原始数据
    # 将空值转为-9999
    x_lai_array_mark[np.isnan(x_lai_array_mark)] = -9999
    # 将数组转换为 float64 类型
    x_lai_array_mark = x_lai_array_mark.astype('float64')
    #### 针对fpar波段将不等于-9999的乘以比例因子0.01；针对lai波段将不等于-9999的乘以比例因子0.1；
    x_lai_array_mark[:, :, 0:1, :, :][x_lai_array_mark[:, :, 0:1, :, :] != -9999] *= 0.01
    x_lai_array_mark[:, :, 1:2, :, :][x_lai_array_mark[:, :, 1:2, :, :] != -9999] *= 0.1
    # 将lai_image_array 里fpar 不在0-1范围内的值赋值为-9999;Fpar 0(min) 100(max) 0.01(比例因子) 因为MCD153H产品nodata被填充为0
    fpar_mask = (x_lai_array_mark[:, :, 0:1, :, :] < 0) | (x_lai_array_mark[:, :, 0:1, :, :] > 1)
    x_lai_array_mark[:, :, 0:1, :, :][fpar_mask] = -9999
    x_lai_array_mark[:, :, 2:3, :, :][fpar_mask] = -9999
    x_lai_array_mark[:, :, 3:, :, :][fpar_mask] = -9999
    #####将x_lai_array中lai波段 不在0-10范围内的值赋值为-9999
    lai_mask = (x_lai_array_mark[:, :, 1:2, :, :] < 0) | (x_lai_array_mark[:, :, 1:2, :, :] > 10)
    x_lai_array_mark[:, :, 1:2, :, :][lai_mask] = -9999
    x_lai_array_mark[:, :, 2:3, :, :][lai_mask] = -9999
    x_lai_array_mark[:, :, 3:, :, :][lai_mask] = -9999
    x_lai_array_mark_qa = x_lai_array_mark[:, :, 2:, :, :]
    lai_qa_array = lai_binary_conversion_and_mark(x_lai_array_mark_qa)
    x_lai_array_mark[:, :, 0, 1, 1]
    # 将对应的mark array与影像合并
    lai_image_array = np.concatenate((x_lai_array_mark[:, :, :2, :, :], lai_qa_array), axis=2)
    lai_image_array[:, :, 2, 1, 1]
    #### 对lai_array进行插值
    lai_image_array_inter = lai_interpolation(lai_image_array)
    lai_image_array_inter[:, 0, 0, 1, 1]
    # 进行各波段填充值统一
    lai_image_array_full = full_filledvalue(lai_image_array_inter)
    return lai_image_array_full


def ref_binary_mask(qa):
    # 原来bit0，3,4,5分别对应于填充值，cloud，cloudshadow，snow，457没有cirrus
    # 由于bit2,14,15 这三个位unused，转换为二进制之后只有13个有效位
    # 因而填充值对应于bit0不变，cloud，cloudshadow，snow向前移一位，应对应于bit2,3,4
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    return bit0 == 0, bit0 == 1


#
def FparLai_QC_binary_mask(qa):
    if qa == -9999:
        return False
    # 将整数转换为8位二进制
    binary_qa = format(int(qa), '08b')
    # 计算各位的值
    bit0 = int(binary_qa[-1])
    bit2 = int(binary_qa[-3])
    bit5 = int(binary_qa[-6])
    bit6 = int(binary_qa[-7])
    bit7 = int(binary_qa[-8])
    # 检查各位是否为1
    return bit0 == 1 or bit2 == 1 or bit5 == 1 or bit6 == 1 or bit7 == 1


def FparExtra_QC_binary_mask(qa):
    if qa == -9999:
        return False
    # 将整数转换为8位二进制
    binary_qa = format(int(qa), '08b')
    # 计算各位的值
    bit2 = int(binary_qa[-3])
    bit4 = int(binary_qa[-5])
    bit5 = int(binary_qa[-6])
    bit6 = int(binary_qa[-7])
    # 检查各位是否为1
    return bit2 == 1 or bit4 == 1 or bit5 == 1 or bit6 == 1


def ref_binary_conversion_and_mark(qa_ref_array):
    # 将不等于-9999的每个值转为二进制
    # 创建新的数组并进行标记
    bit0_0_mask, bit0_1_mask = np.vectorize(ref_binary_mask)(qa_ref_array)
    # print(bit0_0_mask,bit0_1_mask)
    qa_ref_array_mark = np.zeros_like(qa_ref_array)  # 创建一个和 qa_ref_array 相同形状的全0数组
    qa_ref_array_mark[qa_ref_array == -9999] = -9999  # 将对应的位置标记为-9999
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_0_mask] = 0  # 将二进制的bit0为0的位置标记为0
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_1_mask] = 1  # 将二进制的bit0为1的位置标记为1
    return qa_ref_array_mark


def mete_conversion_and_mark(x_mete_array_mark):
    x_mete_array = np.copy(x_mete_array_mark)  # 复制输入数组，以保留原始数据
    # 仅针对太阳辐射变量 将等于是-9999，其余的正常值 将单位从J/m2 转为 W/m2  J/m2 = 累计周期（用秒表示）*W/m2
    x_mete_array[:, :, 0:1, :, :][x_mete_array[:, :, 0:1, :, :] != -9999] /= 86400
    # 仅针对2m处的空气温度 将等于是-9999，其余的正常值 将单位从K 转为 ℃ 1K=°C+273.15
    x_mete_array[:, :, 1:2, :, :][x_mete_array[:, :, 1:2, :, :] != -9999] -= 273.15
    # 体积_土壤_水_层_1,2,3,4  单位从(m3 m-3 ) * mm =  (kg m-2) )  (0 - 7 cm) ,(7 -28 cm) , (28-100 cm),(100-289 cm)
    x_mete_array[:, :, 2:3, :, :][x_mete_array[:, :, 2:3, :, :] != -9999] *= 70
    x_mete_array[:, :, 3:4, :, :][x_mete_array[:, :, 3:4, :, :] != -9999] *= 210
    x_mete_array[:, :, 4:5, :, :][x_mete_array[:, :, 4:5, :, :] != -9999] *= 720
    x_mete_array[:, :, 5:6, :, :][x_mete_array[:, :, 5:6, :, :] != -9999] *= 1890
    # 查看标记后的最大最小值
    return x_mete_array


def mete_time_series_interpolation(array_data, band_index, fillvalue_bounds):
    array_data[array_data == 0] = -9999
    array_data = array_data.astype(float)
    array_data[array_data == -9999] = np.nan
    df_band = pd.DataFrame(array_data)
    df_band_linear = df_band.interpolate(method='linear', axis=1, limit_direction='both')
    df_band_linear = df_band_linear.fillna(-9999)
    array_inter = np.array(df_band_linear)
    return array_inter


def mete_process(x_meteorology_array):
    x_mete_array = x_meteorology_array[:, :, :, :, :]
    x_mete_array_mark = mete_conversion_and_mark(x_mete_array)
    interpolated_array = x_mete_array_mark.copy()

    interpolated_array[:, :, 2:, :, :][interpolated_array[:, :, 2:, :, :] == 1] = -9999
    interpolated_array = full_filledvalue(interpolated_array)
    # 添加一个特征，用于表示待插值的和未插值的
    # 创建一个形状为1802, 366, 1, 3, 3的全零数组
    interpolated_array1 = np.zeros((interpolated_array.shape[0], interpolated_array.shape[1], 1, 3, 3))
    # 标识质量
    # 利用条件索引来修改new_array
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] != -9999] = 0
    interpolated_array1[interpolated_array[:, :, 0:1, :, :] == -9999] = 1
    print(interpolated_array1.shape)
    interpolated_array1[:, :, 0, 1, 1]
    interpolated_array[:, :, 0, 1, 1]
    interpolated_array2 = interpolated_array.copy()
    for band_index in range(interpolated_array2.shape[2]):
        for ii in range(interpolated_array2.shape[3]):
            for jj in range(interpolated_array2.shape[4]):
                array_data = interpolated_array2[:, :, band_index, ii, jj]
                fillvalue_bounds = "extrapolate"
                interpolated_array2[:, :, band_index, ii, jj] = mete_time_series_interpolation(array_data,
                                                                                               band_index,
                                                                                               fillvalue_bounds)
    # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
    for iii in range(interpolated_array2.shape[0]):
        # 对每个3*3窗口进行处理
        for jjj in range(interpolated_array2.shape[1]):
            for mmm in range(interpolated_array2.shape[3]):
                # 对每个3*3窗口进行处理
                for nnn in range(interpolated_array2.shape[4]):
                    if np.any(interpolated_array2[iii, jjj, :, mmm, nnn] == -9999):
                        interpolated_array2[iii, jjj, :, mmm, nnn] = -9999
                    else:
                        continue
    # 将空值用-9999来填充
    interpolated_array2[np.isnan(interpolated_array2)] = -9999
    interpolated_array2 = full_filledvalue(interpolated_array2)
    interpolated_array3 = np.concatenate((interpolated_array2, interpolated_array1), axis=2)
    interpolated_array3 = full_filledvalue(interpolated_array3)
    return interpolated_array3


def spatialtime_preprocess(spatialtime_array):
    # spatialtime_array(lon,lat,year,doy)
    # 将spatialtime_array padding成14个特征（n,366,14）
    # 计算各个维度需要填充的数量
    padding = [(0, 0), (0, 0), (0, 10)]  # 分别表示在第一、二、三个维度上的填充数量
    # 使用np.pad进行填充
    padded_array = np.pad(spatialtime_array, padding, mode='constant', constant_values=-9999)
    # 然后变换特征的顺序 ，参照covariate_array = y_image_array[:, :, [4, 9, 10, 11, 12,3,5,16,17,18, 19, 20, 21,22]]
    # 分别对应DoY，Rg_daily，PotRad_uStar_daily，Tair_f_daily，VPD_f_daily，Year，Month，
    # # Vegetation_Abbreviation(IGBP)，Climate_Class_Abbreviation(Koeppen)，Mean_Average_Precipitation(mm)，
    # # Mean_Average_Temperature(degreesC)，Latitude(degrees)，Longitude(degrees)，Elevation(m)
    # 将第3个特征换到第0个
    # 将第0个特征换到第12个
    # 将第1个特征换到第11个
    # 将第2个特征换到第5个
    # 调整特征顺序
    reordered_array = padded_array[:, :, [3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 0, 1, 13]]
    return reordered_array


def adjust_array_shape(arrays):
    for i, array in enumerate(arrays):
        if array.shape[1] != 366:
            new_shape = list(array.shape)
            new_shape[1] = 366
            new_array = np.zeros(new_shape)  # 创建一个形状为366的全零数组
            if array.shape[1] < 366:
                new_array[:, :array.shape[1], ...] = array  # 将原始数据复制到新数组中
                for j in range(array.shape[1], 366):
                    new_array[:, j, ...] = array[:, -1, ...]  # 将原始的最后一天的数据赋值给新数组的缺失部分
            else:  # 如果原始数据维度大于366，则截取前366天数据
                new_array = array[:, :366, ...]
            arrays[i] = new_array
    return arrays


if __name__ == '__main__':
    # 所有region 存储的根目录 每个国家的文件夹
    region_base_dir = "/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/"

    out_base_dir = '/dat1/Fanbin_data/Vegetation_productivity_prediction/Region_prediction/data/blocks_to_input_npy/'
    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_base_dir)
    nation_path = os.path.join(region_base_dir, 'blocks_variables_match')

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    print(state_list)
    for i in range(48, 49):
        state_name = state_list[i]
        print(state_name)
        state_path = os.path.join(nation_path, state_name)
        output_state_path = os.path.join(out_base_dir, state_name)
        if not os.path.isdir(output_state_path):
            os.makedirs(output_state_path)

        # 进入州下面的block目录
        blocklist = os.listdir(state_path)
        for b in range(4201, len(blocklist)):  # len(blocklist)
            block = blocklist[b]

            output_block_path = os.path.join(output_state_path, block)
            if not os.path.isdir(output_block_path):
                os.makedirs(output_block_path)
            x_ref_array_path = output_block_path + '/' + block + '_ref_input3.npy'
            x_lai_array_path = output_block_path + '/' + block + '_lai_input3.npy'
            x_mete_array_path = output_block_path + '/' + block + '_mete_input3.npy'
            y_image_array_all_path = output_block_path + '/' + block + '_spatialtime_input3.npy'
            # 打开之前保存的地理信息文件
            geoinfo_file_path = os.path.join(output_block_path, f'{block}_geoinfo.txt')
            # 检查四个文件是否存在，若都存在则跳过处理下一个文件
            if all(os.path.exists(path) for path in
                   [x_ref_array_path, x_lai_array_path, x_mete_array_path, y_image_array_all_path, geoinfo_file_path]):
                continue  # 如果其中任何一个文件不存在，则跳过处理下一个文件
            else:
                print(block)
                blocks_path = os.path.join(state_path, block)
                x_ref_array = np.load(blocks_path + '/' + block + '_MCD43A4.npy', allow_pickle=True)
                x_lai_array = np.load(blocks_path + '/' + block + '_MCD15A3H.npy', allow_pickle=True)
                x_meteorology_array = np.load(blocks_path + '/' + block + '_DAILY_AGGR.npy', allow_pickle=True)
                spatialtime_array = np.load(blocks_path + '/' + block + '_spatialtime.npy', allow_pickle=True)

                # # # 对ref变量进行处理
                x_ref_array[:, :, 0, 1, 1]
                ref_image_array = ref_preprocess(x_ref_array)
                # ref_image_array[:, :, 0, 1, 1]
                # check_imagemaxmin(ref_image_array)
                # check_dim_fill(ref_image_array)
                #
                # # # # # 对lai进行处理
                # x_lai_array[:,:,0,1,1]
                lai_image_array = lai_preprocess(x_lai_array)
                lai_image_array[:, :, 0, 1, 1]
                # # check_imagemaxmin(lai_image_array)
                # # check_dim_fill(lai_image_array)
                mete_image_array = mete_process(x_meteorology_array)
                x_meteorology_array[:, :, 0, 1, 1]
                # check_imagemaxmin(mete_image_array)
                # check_dim_fill(mete_image_array)
                # #
                # # 对空间位置数组进行处理
                spatialtime_array_padding = spatialtime_preprocess(spatialtime_array)

                # 将ref_image_array，lai_image_array，mete_image_array和flux_image_array作为输入参数传递给这个函数，并接收返回的数组。
                ref_image_array, lai_image_array, mete_image_array, spatialtime_array_padding = adjust_array_shape(
                    [ref_image_array, lai_image_array, mete_image_array, spatialtime_array_padding])

                print(ref_image_array.shape, lai_image_array.shape, mete_image_array.shape,
                      spatialtime_array_padding.shape)

                # 将处理好的几个变量输出保存
                np.save(output_block_path + '/' + block + '_ref_input3.npy', ref_image_array)
                np.save(output_block_path + '/' + block + '_lai_input3.npy', lai_image_array)
                np.save(output_block_path + '/' + block + '_mete_input3.npy', mete_image_array)
                np.save(output_block_path + '/' + block + '_spatialtime_input3.npy', spatialtime_array_padding)
                print('The ' + block + ' have done')