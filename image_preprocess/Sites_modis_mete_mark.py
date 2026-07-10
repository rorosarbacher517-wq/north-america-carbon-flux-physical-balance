# 在Modis_data_mark 的基础上增加了meteorology 6个变量的标记

import numpy as np

def ref_binary_mask(qa):
    # 原来bit0，3,4,5分别对应于填充值，cloud，cloudshadow，snow，457没有cirrus
    # 由于bit2,14,15 这三个位unused，转换为二进制之后只有13个有效位
    # 因而填充值对应于bit0不变，cloud，cloudshadow，snow向前移一位，应对应于bit2,3,4
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    return bit0 == 0,bit0 ==1
#
# def FparExtra_QC_binary_mask(qa):
#     # 'FparExtra_QC'bit2，4,5,6分别对应于snow，cirrus,cloud，cloudshadow
#     qa_int = int(qa)
#     binary_qa = np.binary_repr(qa_int)
#     binary_qa = int(binary_qa)
#     bit2 = (binary_qa >> 2) & 1
#     bit4 = (binary_qa >> 4) & 1
#     bit5 = (binary_qa >> 5) & 1
#     bit6 = (binary_qa >> 6) & 1
#     return bit2 == 1 or bit4 == 1 or bit5 == 1 or bit6 == 1
#
# def FparLai_QC_binary_mask(qa):
#     # 'FparLai_QC' bit0,2,5,6,7分别对应 bit0:Other quality (back−up algorithm or fill values);
#     # bit2:Dead detectors caused >50% adjacent detector retrieval;
#     # bit5-7:geometry, empirical algorithm,Pixel not produced at all
#     qa_int = int(qa)
#     binary_qa = np.binary_repr(qa_int)
#     binary_qa = int(binary_qa)
#     bit0 = binary_qa & 1
#     bit2 = (binary_qa >> 2) & 1
#     bit5 = (binary_qa >> 5) & 1
#     bit6 = (binary_qa >> 6) & 1
#     bit7 = (binary_qa >> 7) & 1
#     return bit0 == 1 or bit2 == 1 or bit5 == 1 or bit6 == 1 or bit7==1
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

def lai_binary_conversion_and_mark(lai_qa_array):
    # 将不等于-9999的每个值转为二进制
    # 将值转为整数
    # np.binary_repr函数得到的二进制字符串是按正常的方式从左到右表示的，即最高位（最左边）对应最高的权值，最低位（最右边）对应最低的权值。
    # 创建新的数组并进行标记
    FparLai_QC_mask = np.vectorize(FparLai_QC_binary_mask)(lai_qa_array[:,:,0,:,:])
    # FparExtra_QC的值需要分别乘以比例因子 FparStdDev	0(min)	100(max)	0.01(scale)；LaiStdDev 0(min)	100(max)	0.1(scale)
    FparExtra_QC_mask = np.vectorize(FparExtra_QC_binary_mask)(lai_qa_array[:,:,1,:,:])
    # print(FparLai_QC_mask,FparExtra_QC_mask)
    qa_lai_array_mark = np.zeros_like(lai_qa_array[:,:,0,:,:])  # 创建一个和 qa_ref_array 相同形状的全0数组
    qa_lai_array_mark[lai_qa_array[:,:,0,:,:] == -9999] = -9999  # 将对应的位置标记为-9999
    qa_lai_array_mark[(lai_qa_array[:,:,0,:,:] != -9999) & (FparLai_QC_mask | FparExtra_QC_mask)] = 1  # 将视为坏数据的位置标记为1
    qa_lai_array_mark[(lai_qa_array[:,:,0,:,:] != -9999) & ~FparLai_QC_mask & ~FparExtra_QC_mask] = 0  # 将视为好数据的位置标记为0
    # qa_lai_array_mark reshape 为[:,:,1,:,:]
    expanded_qa_lai_array_mark = np.reshape(qa_lai_array_mark,
                                            (qa_lai_array_mark.shape[0], qa_lai_array_mark.shape[1], 1,
                                             qa_lai_array_mark.shape[2], qa_lai_array_mark.shape[3]))
    # 检查改变后的形状
    return expanded_qa_lai_array_mark

def ref_binary_conversion_and_mark(qa_ref_array):
    # 将不等于-9999的每个值转为二进制
    # 创建新的数组并进行标记
    bit0_0_mask,bit0_1_mask = np.vectorize(ref_binary_mask)(qa_ref_array)
    # print(bit0_0_mask,bit0_1_mask)
    qa_ref_array_mark = np.zeros_like(qa_ref_array)  # 创建一个和 qa_ref_array 相同形状的全0数组
    qa_ref_array_mark[qa_ref_array == -9999] = -9999  # 将对应的位置标记为-9999
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_0_mask] = 0  # 将二进制的bit0为0的位置标记为0
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_1_mask] = 1  # 将二进制的bit0为1的位置标记为1
    return qa_ref_array_mark

def mete_conversion_and_mark(x_mete_array_mark):
    x_mete_array = np.copy(x_mete_array_mark)  # 复制输入数组，以保留原始数据
    # 依次查看气象变量的最大最小值,将不在最大最小值范围内以及原本为空值（后被填充为-9999）的值设为-9999，
    # 遍历每个波段
    # for i in range(6):
    #     band_values = x_mete_array[:, :, i, :, :]
    #     # 过滤非有限数值
    #     finite_values = band_values[np.isfinite(band_values)] # 因为有值为无穷大inf
    #     # 计算最大最小值
    #     max_value = np.nanmax(finite_values)
    #     min_value = np.nanmin(finite_values)
    #     # 将不在最大最小值范围内的值设为-9999
    #     band_values[(band_values < min_value) | (band_values > max_value)] = -9999
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

def check_imagemaxmin(x_iamges_array):
    for i in range(x_iamges_array.shape[2]):
        band_values = x_iamges_array[:, :, i, :, :]  # 选择特定的波长数组
        finite_values_after_replacement = band_values[band_values != -9999]
        max_values_after_replacement = np.nanmax(finite_values_after_replacement)
        min_values_after_replacement = np.nanmin(finite_values_after_replacement)
        print(f"波段 {i + 1} 除去-9999以外的最大值为: {max_values_after_replacement}, 最小值为: {min_values_after_replacement}")

def check_dim_fill(x_image_array):
    for bi in range(x_image_array.shape[2]):
        filled_n = (x_image_array[:, :, bi, :, :] != -9999).sum()
        print("band" + str(bi) + "    filled_n" + str(filled_n))

# 给定影像的根路径
if __name__ == "__main__":
    # download the data
    inputdata_path = 'E:/Paper code/code/Carbon_flux_estimation/data/sites_inputdata/'
    x_ref_array = np.load(inputdata_path + 'x_ref_modis_new.npy', allow_pickle=True)
    x_lai_array0 = np.load(inputdata_path + 'x_lai_modis_new.npy', allow_pickle=True)
    x_meteorology_array = np.load(inputdata_path + 'x_mete_era5_new.npy', allow_pickle=True)
    y_meteorological_array = np.load(inputdata_path + 'y_flux_meteorological_new.npy', allow_pickle=True)
    # check_dim_fill(x_lai_array)
    x_lai_array0[:,:,2,1,1]
    y_meteorological_array[:,:,0]
    x_meteorology_array[:, :, 0, 1, 1]
    x_ref_array[:,:,13,1,1]
    y_meteorological_array[:,0,:]
    check_imagemaxmin(x_lai_array0)
    # print(x_ref_array.shape)
    # print(x_lai_array.shape)
    # print(y_meteorological_array.shape)
    # # 根据QA波段分别标识反射率波段和lai波段的质量
    # # 反射率是前7个波段是反射率 后7个波段是反射率对应的QA波段；lai是前两个波段分别是Fpar,lai, 'FparLai_QC', 'FparExtra_QC'
    # # 将反射率质量好的标记为0；质量差的标记为1(56, 366, 14, 3, 3),下载的原始影像值为空的标记为-9999
    # ####将ref_image_array所有波段里  反射率值不在0-3.2766范围内的像素值赋值为-9999。因为mcd43a4的最大\最小值即为32766\0,比例因子为0.0001, 将不在范围内的值赋值为-9999
    ref_mask1 = (x_ref_array[:, :, :7, :, :] < 0) | (x_ref_array[:, :, :7, :, :] > 3.2766)
    ref_mask2 = (x_ref_array[:, :, 7:, :, :] < 0)
    x_ref_array[:, :, :7, :, :][ref_mask1] = -9999
    x_ref_array[:, :, 7:, :, :][ref_mask1] = -9999
    x_ref_array[:, :, :7, :, :][ref_mask2] = -9999
    x_ref_array[:, :, 7:, :, :][ref_mask2] = -9999
    check_imagemaxmin(x_ref_array)
    x_ref_array[:,:,7,1,1]
    ref_qa_array = x_ref_array[:,:,7:,:,:]
    ref_qa_array[:,:,0,1,1]
    ref_qa_array_mark = ref_binary_conversion_and_mark(ref_qa_array)
    ref_qa_array_mark[:,:,0,1,1]

    # ####将x_lai_array中fpar波段 不在0-1范围内的值赋值为-9999
    # ## 将lai_image_array 里fpar 不在0-1范围内的值赋值为-9999;Fpar 0(min) 100(max) 0.01(比例因子)
    ## x_lai_array[:, :, 0:1, :, :] /= 10
    ## x_lai_array[:, :, 1:2, :, :] *= 10
    #
    # # # 将lai_image_array 里fpar 不在0-1范围内的值赋值为-9999;Fpar 0(min) 100(max) 0.01(比例因子) 因为MCD153H产品nodata被填充为0
    x_lai_array = x_lai_array0
    fpar_mask = (x_lai_array[:, :, 0:1, :, :] < 0) | (x_lai_array[:, :, 0:1, :, :] > 1)
    x_lai_array[:, :, 0:1, :, :][fpar_mask] = -9999
    x_lai_array[:, :, 2:3, :, :][fpar_mask] = -9999
    x_lai_array[:, :, 3:, :, :][fpar_mask] = -9999
    #####将x_lai_array中lai波段 不在0-10范围内的值赋值为-9999
    lai_mask = (x_lai_array[:, :, 1:2, :, :] < 0) | (x_lai_array[:, :, 1:2, :, :] > 10)
    x_lai_array[:, :, 1:2, :, :][lai_mask] = -9999
    x_lai_array[:, :, 2:3, :, :][lai_mask] = -9999
    x_lai_array[:, :, 3:, :, :][lai_mask] = -9999
    x_lai_array[:, :, 1, 1, 1]
    lai_qa_array = x_lai_array[:,:,2:,:,:]
    lai_qa_array_mark = lai_binary_conversion_and_mark(lai_qa_array)
    lai_qa_array_mark[:, :, 0, 1, 1]
    x_lai_array[:, :, 2, 1, 1]
    lai_image_array = np.concatenate((x_lai_array[:, :, :2, :, :], lai_qa_array_mark), axis=2)
    # check_imagemaxmin(lai_image_array)

    x_meteorology_array[:,:,0,1,1]
    x_mete_array = x_meteorology_array[:, :, :, :, :]
    x_mete_array_mark = mete_conversion_and_mark(x_mete_array)
    x_mete_array_mark[:,:,0,1,1]
    # check_imagemaxmin(x_mete_array_mark)
    # # # ## 将对应的mark array与影像合并
    ref_image_array = np.concatenate((x_ref_array[:,:,:7,:,:], ref_qa_array_mark), axis=2)
    lai_image_array = np.concatenate((x_lai_array[:, :, :2, :, :], lai_qa_array_mark), axis=2)
    mete_image_array = x_mete_array_mark
    ref_qa_array_mark[:, :, 0, 1, 1]
    ref_image_array[:, :, 7, 1, 1]
    # print(ref_image_array.shape,lai_image_array.shape,mete_image_array.shape,y_meteorological_array.shape)
    # check_imagemaxmin(ref_image_array)
    check_imagemaxmin(lai_image_array)
    # check_imagemaxmin(mete_image_array)
    # check_dim_fill(x_lai_array[:, :, :2, :, :])
    check_dim_fill(lai_image_array)
    outputdata_path = 'D:/Carbon_flux/data/sites_images_mark/'
    np.save(outputdata_path + 'x_ref_modis_new_mark.npy', ref_image_array)
    np.save(outputdata_path + 'x_lai_modis_new_mark.npy', lai_image_array)
    np.save(outputdata_path + 'x_mete_era5_mark.npy', mete_image_array)
    np.save(outputdata_path + 'y_flux_meteorological_mark.npy', y_meteorological_array)
    # #
    # # # #
    # # #
