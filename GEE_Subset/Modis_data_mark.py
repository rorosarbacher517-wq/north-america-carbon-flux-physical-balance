#
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

def FparExtra_QC_binary_mask(qa):
    # 'FparExtra_QC'bit2，4,5,6分别对应于snow，cirrus,cloud，cloudshadow
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit2 = (binary_qa >> 2) & 1
    bit4 = (binary_qa >> 4) & 1
    bit5 = (binary_qa >> 5) & 1
    bit6 = (binary_qa >> 6) & 1
    return bit2 == 1 or bit4 == 1 or bit5 == 1 or bit6 == 1

def FparLai_QC_binary_mask(qa):
    # 'FparLai_QC' bit0，2,5,6,7分别对应 bit0:Other quality (back−up algorithm or fill values);
    # bit2:Dead detectors caused >50% adjacent detector retrieval;
    # bit5-7:geometry, empirical algorithm,Pixel not produced at all
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    bit2 = (binary_qa >> 2) & 1
    bit5 = (binary_qa >> 5) & 1
    bit6 = (binary_qa >> 6) & 1
    bit7 = (binary_qa >> 7) & 1
    return bit0 == 1 or bit2 == 1 or bit5 == 1 or bit6 == 1 or bit7==1

def lai_binary_conversion_and_mark(lai_qa_array):
    # 将不等于-9999的每个值转为二进制
    # 将值转为整数
    # np.binary_repr函数得到的二进制字符串是按正常的方式从左到右表示的，即最高位（最左边）对应最高的权值，最低位（最右边）对应最低的权值。
    # 创建新的数组并进行标记
    FparLai_QC_mask = np.vectorize(FparLai_QC_binary_mask)(lai_qa_array[:,:,0,:,:])
    FparExtra_QC_mask = np.vectorize(FparExtra_QC_binary_mask)(lai_qa_array[:,:,1,:,:])
    print(FparLai_QC_mask,FparExtra_QC_mask)
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
    print(bit0_0_mask,bit0_1_mask)
    qa_ref_array_mark = np.zeros_like(qa_ref_array)  # 创建一个和 qa_ref_array 相同形状的全0数组
    qa_ref_array_mark[qa_ref_array == -9999] = -9999  # 将对应的位置标记为-9999
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_0_mask] = 0  # 将二进制的bit0为0的位置标记为0
    qa_ref_array_mark[(qa_ref_array != -9999) & bit0_1_mask] = 1  # 将二进制的bit0为1的位置标记为1
    return qa_ref_array_mark


# 给定影像的根路径
if __name__ == "__main__":
    # download the data
    inputdata_path = 'E:/Carbon_flux/data/input_data/TEST/'
    x_image_array = np.load(inputdata_path + 'x_image_array_modis_0_270.npy', allow_pickle=True)
    x_lai_array = np.load(inputdata_path + 'x_lai_array_modis_0_270.npy', allow_pickle=True)
    y_meteorological_array = np.load(inputdata_path + 'y_meteorological_array_flux_0_270.npy', allow_pickle=True)

    print(x_image_array.shape)
    print(x_lai_array.shape)
    print(y_meteorological_array.shape)
    # 根据QA波段分别标识反射率波段和lai波段的质量
    # 反射率是前7个波段是反射率 后7个波段是反射率对应的QA波段；lai是前两个波段分别是Fpar,lai, 'FparLai_QC', 'FparExtra_QC'
    # 将反射率质量好的标记为0；质量差的标记为1 (56, 366, 14, 3, 3),下载的原始影像值为空的标记为-9999
    ref_qa_array = x_image_array[:, :, 7:, :, :]
    ref_qa_array_mark = ref_binary_conversion_and_mark(ref_qa_array)
    print(ref_qa_array_mark)
    lai_qa_array = x_lai_array[:, :, 2:, :, :]
    lai_qa_array_mark = lai_binary_conversion_and_mark(lai_qa_array)
    print(lai_qa_array_mark)
    # 将ref_image_array前7个波段里 不在0-3.2766范围内的值赋值为-9999
    # 找出不在0-3.2766范围内的值的索引
    out_of_range_idx = np.where((x_image_array[:, :, :7, :, :] < 0) | (x_image_array[:, :, :7, :, :] > 3.2766))
    # 将不在范围内的值赋值为-9999
    x_image_array[out_of_range_idx] = -9999

    ## 将对应的mark array与影像合并
    ref_image_array = np.concatenate((x_image_array[:, :, :7, :, :], ref_qa_array_mark), axis=2)
    lai_image_array = np.concatenate((x_lai_array[:, :, 1:2, :, :], lai_qa_array_mark), axis=2)
    print(ref_image_array.shape, lai_image_array.shape, y_meteorological_array.shape)

    np.save(inputdata_path + 'x_image_array_modis_0_270_mark_v1.npy', ref_image_array)
    np.save(inputdata_path + 'x_lai_array_modis_0_270_mark_v1.npy', lai_image_array)
    np.save(inputdata_path + 'y_meteorological_array_flux_0_270_mark_v1.npy', y_meteorological_array)





