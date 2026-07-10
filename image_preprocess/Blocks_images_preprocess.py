import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator

def ref_binary_mask(qa):
    # 原来bit0，3,4,5分别对应于填充值，cloud，cloudshadow，snow，457没有cirrus
    # 由于bit2,14,15 这三个位unused，转换为二进制之后只有13个有效位
    # 因而填充值对应于bit0不变，cloud，cloudshadow，snow向前移一位，应对应于bit2,3,4
    qa_int = int(qa)
    binary_qa = np.binary_repr(qa_int)
    binary_qa = int(binary_qa)
    bit0 = binary_qa & 1
    return bit0 == 0,bit0 ==1

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

# # # 通过3*3窗口内的有效值进行线性插值
def ref_interpolate_window(interpolated_b_array):
    for b in range(interpolated_b_array.shape[0]):
        if b >= 0 and b <= 6:
            bb = b + 7
        else:
            bb = b
        window = interpolated_b_array[bb, :, :]
        window[window != 0] = -9999  # Convert non-zero values to -9999
        window[window == -9999] = np.nan  # Convert -9999 to NaN

        nan_indices = np.argwhere(np.isnan(window))  # Get the indices of NaN values
        non_nan_indices = np.argwhere(~np.isnan(window))  # Get the indices of non-NaN values

        if len(non_nan_indices) > 1 and len(non_nan_indices) < 9:
            x = non_nan_indices[:, 1]
            y = non_nan_indices[:, 0]
            values = interpolated_b_array[b, y, x].reshape(-1, 1)
            non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)

            interp = NearestNDInterpolator(non_nan_indices_reshaped, values)  # Create nearest neighbor interpolator
            interpolated_values = interp(nan_indices)  # Perform nearest neighbor interpolation for NaN values

            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:,
                                                       1]] = interpolated_values.flatten()  # Assign the interpolated values to the NaN positions
        elif len(non_nan_indices) == 1:
            non_nan_value = interpolated_b_array[b, non_nan_indices[0, 0], non_nan_indices[0, 1]]
            interpolated_b_array[
                b, nan_indices[:, 0], nan_indices[:, 1]] = non_nan_value  # Replace NaN values with the non-NaN value
        elif len(non_nan_indices) == 9:
            interpolated_b_array[b, :, :] = interpolated_b_array[b, :, :]
    return interpolated_b_array

#
def ref_interpolation(ref_image_array):
    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = ref_image_array.copy()
    #####  针对每个波段，每一天的3*3窗口内的值进行插值
    # 应该先把一个时间步长3*3窗口内的低质量值用高质量值进行替换，这样就只有高质量值和-9999了
    # 对每个时间步长进行处理
    for i in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for j in range(interpolated_array.shape[1]):
            interpolated_array[i, j, :, :, :] = ref_interpolate_window(interpolated_array[i, j, :, :, :])

    # check_imagemaxmin(interpolated_array)
    interpolated_array[:, :, 7:, :, :][interpolated_array[:, :, 7:, :, :] == 1] = -9999
    # 对每个波段进行每一年的时间序列插值"
    for yy in range(interpolated_array.shape[0]):
        for bb in range(interpolated_array.shape[2]):
            for jj in range(interpolated_array.shape[3]):  # 遍历第二个维度
                for kk in range(interpolated_array.shape[4]):  # 遍历第三个维度
                    mark_values = interpolated_array[yy, :, bb, jj, kk]
                    # if (0 in mark_values) and (1 in mark_values):
                    # 创建一个索引数组，包含所有非-9999和坏数据值的索引
                    # mark_valid_indices = np.where(mark_values == 0)[0]
                    mark_valid_indices = np.where(mark_values != -9999)[0]
                    # 待插值的数据索引,质量标识为1 低质量像素
                    interpolate_indices = np.where(interpolated_array[yy, :, bb, jj, kk] == -9999)[0]
                    if len(mark_valid_indices) > 1 and len(mark_valid_indices) < 366:
                        if bb < 7:
                            fillvalue_bounds = (0, 1)
                        else:
                            fillvalue_bounds = "extrapolate"
                        # 对于只有一条有效记录的数据，线性插值没有意义，不需要进行插值
                        # 这里需要针对每个波段分别建立插值函数
                        f = interpolate.interp1d(mark_valid_indices,
                                                 interpolated_array[yy, :, bb, jj, kk][mark_valid_indices],
                                                 kind='linear', bounds_error=False,
                                                 fill_value=fillvalue_bounds)

                        interpolated_values = f(np.arange(len(interpolate_indices)))
                        # 使用索引数组进行插值，填充数据中的空值
                        # interpolated_values = f(np.arange(len(interpolated_array[yy, :, ii, jj, kk])))
                        # 使用最邻近的有效值进行填充
                        # for index in range(len(interpolate_indices)):
                        #     if abs(interpolated_values[index] - fillvalue_bounds[0]) < 1e-6 or abs(
                        #             interpolated_values[index] - fillvalue_bounds[1]) < 1e-6:
                        #         nearest_valid_index = mark_valid_indices[
                        #             np.argmin(abs(mark_valid_indices - interpolate_indices[index]))]
                        #         interpolated_values[index] = interpolated_array[yy, :, bb, jj, kk][nearest_valid_index]
                        # 将插值的结果保存到新数组中
                        interpolated_array[yy, :, bb, jj, kk][interpolate_indices] = interpolated_values
                    elif len(mark_valid_indices) == 1:
                        # valid_value = mark_values[mark_valid_indices[0]]
                        interpolated_array[yy, :, bb, jj, kk][interpolate_indices] = \
                            interpolated_array[yy, :, bb, jj, kk][mark_valid_indices[0]]
                        # 如果只有一条有效记录的数据，将所有的数值设为该有效记录的值
                    else:
                        # Set all the days of the year to -9999
                        interpolated_array[yy, :, bb, jj, kk] = interpolated_array[yy, :, bb, jj, kk]
    interpolated_array[:, :, 3, 1, 1]
    return interpolated_array

def full_filledvalue(interpolated_array):
    # 插值完之后 ，将mark标记仍为1的视为-9999  这种情况是由于整个年时间序列366天当中没有一个像素是质量好的，所以仍会出现质量为1的情况
    # 若某一天的某个像素某个波段为-9999，而其他波段不是-9999，则把这一天这个像素的所有波段值均设置为-9999
    for iii in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for jjj in range(interpolated_array.shape[1]):
            for mmm in range(interpolated_array.shape[3]):
                # 对每个3*3窗口进行处理
                for nnn in range(interpolated_array.shape[4]):
                    if np.any(interpolated_array[iii, jjj, :, mmm, nnn] == -9999):
                        interpolated_array[iii, jjj, :, mmm, nnn] = -9999
                    else:
                        continue
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

    # 将质量标识仍为1的转为-9999
    ref_image_array_inter[:, :, 7:, :, :][ref_image_array_inter[:, :, 7:, :, :] == 1] = -9999
    #### 检查是否有波段填充值数量不一致的情况 若有则将该像素所有的波段都视为填充值
    ref_image_array_full  = full_filledvalue(ref_image_array_inter)
    return ref_image_array_full

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

# 通过3*3窗口内的有效值进行线性插值
def lai_interpolate_window(window, interpolated_b_array):
    # Convert non-zero values to -9999
    window[window != 0] = -9999
    # 将window数组转换为浮点数类型
    window = window.astype(float)
    # Convert values of -9999 to NaN
    window[window == -9999] = np.nan

    # Get the indices of NaN and non-NaN values
    nan_indices = np.argwhere(np.isnan(window))
    non_nan_indices = np.argwhere(~np.isnan(window))

    for b in range(interpolated_b_array.shape[0]):
        # For windows with more than one non-NaN value, use nearest neighbor interpolation
        if len(non_nan_indices) > 1 and len(non_nan_indices) < 9:
            x = non_nan_indices[:, 1]
            y = non_nan_indices[:, 0]
            values = interpolated_b_array[b, y, x]
            values_reshaped = values.reshape(-1, 1)
            non_nan_indices_reshaped = non_nan_indices.reshape(-1, 2)

            # Create a nearest neighbor interpolator
            interp = NearestNDInterpolator(non_nan_indices_reshaped, values_reshaped)

            # Perform nearest neighbor interpolation for NaN values
            interpolated_values = interp(nan_indices)

            # Assign the interpolated values to the NaN positions in the current window
            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values.flatten()

        # For windows with only one non-NaN value, replace NaN values with the non-NaN value
        elif len(non_nan_indices) == 1:
            non_nan_value = interpolated_b_array[b, non_nan_indices[0, 0], non_nan_indices[0, 1]]
            interpolated_b_array[b, nan_indices[:, 0], nan_indices[:, 1]] = non_nan_value
        elif len(non_nan_indices) == 9:
            interpolated_b_array[b, :, :] = interpolated_b_array[b, :, :]
    return interpolated_b_array

def lai_interpolation(fpar_lai_array):
    # 创建一个新的数组用于储存插值后的结果
    interpolated_array = fpar_lai_array.copy()
    #####  针对每个波段，每一天的3*3窗口内的值进行插值
    # 应该先把一个时间步长3*3窗口内的低质量值用高质量值进行替换，这样就只有高质量值和-9999了
    # 对每个时间步长进行处理
    for i in range(interpolated_array.shape[0]):
        # 对每个3*3窗口进行处理
        for j in range(interpolated_array.shape[1]):
            # 提取当前质量标识的窗口
            window = interpolated_array[i, j, 2, :, :]
            # 对每个波段，每个窗口内的值进行插值，用窗口内的有效值来插值代替nan值
            # 确认窗口内是否存在nan值并进行插值
            if (0 in window) and ((1 in window) or (-9999 in window)):
                interpolated_array[i, j, :, :, :] = lai_interpolate_window(window, interpolated_array[i, j, :, :, :])
            else:
                interpolated_array[i, j, :, :, :] = interpolated_array[i, j, :, :, :]
    interpolated_array[10, 5, 0, :, :]
    # 插值完之后 ，将mark标记仍为1的视为-9999
    interpolated_array[:, :, 2:, :, :][interpolated_array[:, :, 2:, :, :] == 1] = -9999
    interpolated_array  = full_filledvalue(interpolated_array)

    # 对每个波段进行每一年的时间序列插值"
    for yy in range(interpolated_array.shape[0]):
        for ii in range(interpolated_array.shape[2]):
            for jj in range(interpolated_array.shape[3]):  # 遍历第二个维度
                for kk in range(interpolated_array.shape[4]):  # 遍历第三个维度
                    win_mark_values = interpolated_array[yy, :, 2, jj, kk]
                    band_mark_values = interpolated_array[yy, :, ii, jj, kk]
                    # 创建一个索引数组，包含所有非-9999和坏数据值的索引
                    band_mark_valid_indices = np.where((win_mark_values != -9999) & (band_mark_values != 0))[0]
                    quality_mark_valid_indices = np.where(win_mark_values != -9999)[0]
                    # 待插值的数据索引,质量标识为1 低质量像素
                    # 待插值的数据索引,将等于-9999和0的像素视为待插值像素，不等于的视为好像素
                    band_interpolate_indices = \
                    np.where((interpolated_array[yy, :, 2, jj, kk] == -9999) | (interpolated_array[yy, :, ii, jj, kk] == 0)| (interpolated_array[yy, :, ii, jj, kk] == -9999))[0]
                    quality_interpolate_indices = np.where(interpolated_array[yy, :, 2, jj, kk] == -9999)[0]
                    if len(quality_mark_valid_indices) > 1 and len(quality_mark_valid_indices) < 366:
                        if ii == 0:
                            fillvalue_bounds = (0.0, 1.0)
                            mark_valid_indices = band_mark_valid_indices
                            interpolate_indices = band_interpolate_indices
                        elif ii == 1:
                            fillvalue_bounds = (0.0, 10.0)
                            mark_valid_indices = band_mark_valid_indices
                            interpolate_indices = band_interpolate_indices
                        else:
                            fillvalue_bounds = "extrapolate"
                            mark_valid_indices = quality_mark_valid_indices
                            interpolate_indices = quality_interpolate_indices
                        # 对于只有一条有效记录的数据，线性插值没有意义，不需要进行插值
                        f = interpolate.interp1d(mark_valid_indices,
                                                 interpolated_array[yy, :, ii, jj, kk][mark_valid_indices],
                                                 kind='linear', bounds_error=False, fill_value=fillvalue_bounds)
                        interpolated_values = f(np.arange(len(interpolate_indices)))
                        interpolated_array[yy, :, ii, jj, kk][interpolate_indices] = interpolated_values
                    elif len(mark_valid_indices) == 1:
                        # valid_value = mark_values[mark_valid_indices[0]]
                        interpolated_array[yy, :, ii, jj, kk][interpolate_indices] = \
                        interpolated_array[yy, :, ii, jj, kk][mark_valid_indices[0]]
                        # 如果只有一条有效记录的数据，将所有的数值设为该有效记录的值
                    else:
                        # Set all the days of the year to -9999
                        interpolated_array[yy, :, ii, jj, kk] = interpolated_array[yy, :, ii, jj, kk]
    interpolated_array[:, :, 2, 0, 1]
    # check_imagemaxmin(interpolated_array)
    # 保存插值后的数据
    # 插值完之后 ，将mark标记仍为1的视为-9999
    interpolated_array[:, :, 1:2, :, :][interpolated_array[:, :, 1:2, :, :] == 1] = -9999
    return interpolated_array

def lai_preprocess(x_lai_array):
    x_lai_array[:,:,0,1,1]
    x_lai_array_mark = np.copy(x_lai_array)  # 复制输入数组，以保留原始数据
    # 将空值转为-9999
    x_lai_array_mark[np.isnan(x_lai_array_mark)] =  -9999
    # 将数组转换为 float64 类型
    x_lai_array_mark = x_lai_array_mark.astype('float64')
    #### 针对fpar波段将不等于-9999的乘以比例因子0.01；针对lai波段将不等于-9999的乘以比例因子0.1；
    x_lai_array_mark[:, :, 0:1, :, :][x_lai_array_mark[:, :, 0:1, :, :] != -9999] *= 0.01
    x_lai_array_mark[:, :, 1:2, :, :][x_lai_array_mark[:, :, 1:2, :, :] != -9999] *= 0.1
    # 将lai_image_array 里fpar 不在0-1范围内的值赋值为-9999;Fpar 0(min) 100(max) 0.01(比例因子) 因为MCD153H产品nodata被填充为0
    fpar_mask = (x_lai_array_mark[:, :, 0:1, :, :] < 0) | (x_lai_array_mark[:, :, 0:1, :, :] > 1)| (x_lai_array_mark[:, :, 0:1, :, :] == 0)
    x_lai_array_mark[:, :, 0:1, :, :][fpar_mask] = -9999
    x_lai_array_mark[:, :, 2:3, :, :][fpar_mask] = -9999
    x_lai_array_mark[:, :, 3:, :, :][fpar_mask] = -9999
    #####将x_lai_array中lai波段 不在0-10范围内的值赋值为-9999
    lai_mask = (x_lai_array_mark[:, :, 1:2, :, :] < 0) | (x_lai_array_mark[:, :, 1:2, :, :] > 10)| (x_lai_array_mark[:, :, 1:2, :, :] == 0)
    x_lai_array_mark[:, :, 1:2, :, :][lai_mask] = -9999
    x_lai_array_mark[:, :, 2:3, :, :][lai_mask] = -9999
    x_lai_array_mark[:, :, 3:, :, :][lai_mask] = -9999

    x_lai_array_mark_qa = x_lai_array_mark[:, :, 2:, :, :]
    lai_qa_array = lai_binary_conversion_and_mark(x_lai_array_mark_qa)
    x_lai_array_mark_qa[:,:,0,1,1]
    # 将对应的mark array与影像合并
    lai_image_array = np.concatenate((x_lai_array_mark[:, :, :2, :, :], lai_qa_array), axis=2)

    #### 对lai_array进行插值
    lai_image_array_inter = lai_interpolation(lai_image_array)
    lai_image_array_inter[:, :, 0, 1, 1]
    # 对于超出范围的值进行近邻替换
    # # 仅针对前两个波段的数组进行处理
    # interpolated_array_subset = lai_image_array_inter[:, :, :2, :, :]
    # # 替换值为0的像素为最邻近的非0值或-9999值
    # for b in range(interpolated_array_subset.shape[2]):  # 针对每个波段
    #     for i in range(interpolated_array_subset.shape[3]):  # 遍历每个像素位置的第一个维度
    #         for j in range(interpolated_array_subset.shape[4]):  # 遍历每个像素位置的第二维度
    #             if np.any(interpolated_array_subset[:, :, b, i, j] == 0):  # 如果像素值为0
    #                 # 找到最邻近的非0值或-9999值
    #                 nearest_values = [value for value in np.unique(interpolated_array_subset[:, :, b, i, j]) if
    #                                   value != 0 and value != -9999]
    #                 nearest_value = min(nearest_values,key=lambda x: abs(x - 0)) if nearest_values else -9999  # 找到最邻近的值
    #                 # 用最邻近的非0值或-9999值替换0值
    #                 interpolated_array_subset[:, :, b, i, j] = np.where(interpolated_array_subset[:, :, b, i, j] == 0,
    #                                                                     nearest_value,
    #                                                                     interpolated_array_subset[:, :, b, i, j])
    #             else:
    #                 # 保持原值
    #                 pass
    #### 检查是否有波段填充值数量不一致的情况 若有则将该像素所有的波段都视为填充值
    lai_image_array_full  = full_filledvalue(lai_image_array_inter)
    return lai_image_array_full

def mete_conversion_and_mark(x_mete_array_mark):
    x_mete_array = np.copy(x_mete_array_mark)  # 复制输入数组，以保留原始数据
    # 将x_mete_array中的空值填充为-9999
    x_mete_array[np.isnan(x_mete_array)] =  -9999
    # 依次查看气象变量的最大最小值,将不在最大最小值范围内以及原本为空值（后被填充为-9999）的值设为-9999，
    # 仅针对太阳辐射变量 将等于是-9999，其余的正常值 将单位从J/m2 转为 W/m2  J/m2 = 累计周期（用秒表示）*W/m2
    x_mete_array[:, :, 0:1, :, :][x_mete_array[:, :, 0:1, :, :] != -9999] /= 86400
    solar_mask = (x_mete_array[:, :, 0:1, :, :] < 0) | (x_mete_array[:, :, 0:1, :, :] > 500)
    x_mete_array[:, :, 0:1, :, :][solar_mask] = -9999

    # 仅针对2m处的空气温度 将等于是-9999，其余的正常值 将单位从K 转为 ℃ 1K=°C+273.15
    x_mete_array[:, :, 1:2, :, :][x_mete_array[:, :, 1:2, :, :] != -9999] -= 273.15
    temperature_mask = (x_mete_array[:, :, 1:2, :, :] < -50) | (x_mete_array[:, :, 1:2, :, :] > 50)
    x_mete_array[:, :, 1:2, :, :][temperature_mask] = -9999

    # 体积_土壤_水_层_1,2,3,4  单位从(m3 m-3 ) ==> *(10/7)/1000 (kg m-2) )
    x_mete_array[:, :, 2:3, :, :][x_mete_array[:, :, 2:3, :, :] != -9999] *= 70
    x_mete_array[:, :, 3:4, :, :][x_mete_array[:, :, 3:4, :, :] != -9999] *= 210
    x_mete_array[:, :, 4:5, :, :][x_mete_array[:, :, 4:5, :, :] != -9999] *= 720
    x_mete_array[:, :, 5:6, :, :][x_mete_array[:, :, 5:6, :, :] != -9999] *= 1890
    #### 检查是否有波段填充值数量不一致的情况 若有则将该像素所有的波段都视为填充值
    mete_image_array_full = full_filledvalue(x_mete_array)
    return mete_image_array_full

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
    reordered_array = padded_array[:, :, [3, 4, 5, 6, 7,2, 8, 9, 10, 11, 12, 0, 1,13]]
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
    region_base_dir = 'E:/Paper code/code/Vegetation_productivity_prediction/Region_prediction/data'

    out_base_dir = 'E:/Paper code/code/Vegetation_productivity_prediction/Region_prediction/data/blocks_to_input_npy/'
    # 进入国家级下面的州级目录
    nation_list = os.listdir(region_base_dir)
    nation_path = os.path.join(region_base_dir, 'blocks_variables_match')

    # 进入国家级下面的州级目录
    state_list = os.listdir(nation_path)
    state_path = os.path.join(nation_path, state_list[4])

    output_state_path = os.path.join(out_base_dir, state_list[4])
    if not os.path.isdir(output_state_path):
        os.makedirs(output_state_path)

    # 进入州下面的block目录
    # blocklist = os.listdir(state_path)
    blocklist = ['block_0_0','block_0_1','block_0_2', 'block_0_3', 'block_0_4', 'block_0_5','block_0_6',
                 'block_1_0','block_1_1','block_1_2', 'block_1_3', 'block_1_4', 'block_1_5','block_1_6']
    for b in range(0,len(blocklist)): #len(blocklist)
        block = blocklist[b]
        output_block_path = os.path.join(output_state_path, block)
        if not os.path.isdir(output_block_path):
            os.makedirs(output_block_path)

        blocks_path = os.path.join(state_path,block)
        x_ref_array = np.load(blocks_path +'/'+block+'_MCD43A4.npy', allow_pickle=True)
        x_lai_array = np.load(blocks_path +'/'+block+'_MCD15A3H.npy',  allow_pickle=True)
        x_meteorology_array = np.load(blocks_path+'/'+block+'_DAILY_AGGR.npy', allow_pickle=True)
        spatialtime_array = np.load(blocks_path +'/'+block+'_spatialtime.npy', allow_pickle=True)

        # 对ref变量进行处理
        ref_image_array = ref_preprocess(x_ref_array)

        # check_imagemaxmin(ref_image_array)
        # check_dim_fill(ref_image_array)

        # 对lai进行处理
        x_lai_array[:,:,0,1,1]
        lai_image_array = lai_preprocess(x_lai_array)
        lai_image_array[:,:,0,1,1]
        # check_imagemaxmin(lai_image_array)
        # check_dim_fill(lai_image_array)

        # 对mete进行处理
        x_mete_array = x_meteorology_array[:, :, :, :, :]
        x_mete_array_mark = mete_conversion_and_mark(x_mete_array)
        #### 检查是否有波段填充值数量不一致的情况 若有则将该像素所有的波段都视为填充值
        mete_image_array_full = full_filledvalue(x_mete_array_mark)
        mete_image_array = mete_image_array_full
        x_meteorology_array[:,:,0,1,1]
        # check_imagemaxmin(mete_image_array)
        # check_dim_fill(mete_image_array)
        #
        # 对空间位置数组进行处理
        spatialtime_array_padding = spatialtime_preprocess(spatialtime_array)

        # 将ref_image_array，lai_image_array，mete_image_array和flux_image_array作为输入参数传递给这个函数，并接收返回的数组。
        ref_image_array, lai_image_array, mete_image_array, spatialtime_array_padding = adjust_array_shape(
            [ref_image_array, lai_image_array, mete_image_array, spatialtime_array_padding])

        # print(ref_image_array.shape, lai_image_array.shape, mete_image_array.shape, spatialtime_array_padding.shape)

        # 将处理好的几个变量输出保存
        np.save(output_block_path+'/'+block+'_ref_input1.npy', ref_image_array)
        np.save(output_block_path+'/'+block+'_lai_input1.npy', lai_image_array)
        np.save(output_block_path+'/'+block+'_mete_input1.npy', mete_image_array)
        np.save(output_block_path+'/'+block+'_spatialtime_input1.npy', spatialtime_array_padding)
        print('The ' + block + ' have done')