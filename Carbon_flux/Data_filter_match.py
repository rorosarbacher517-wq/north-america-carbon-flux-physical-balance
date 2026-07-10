

# 本版本是针对modis 反射率lai,气象数据 和flux同时匹配并 进行有无x/y版本的判断 类似于之前的v4 v5 v6 v7

import numpy as np
import numpy as np

from functools import reduce

# 定义一个函数求取三个变量四分位数3倍的上下限值，以此来作为有效数据的阈值
def interquartile_range(y_image_array_all):
    # 创建一个布尔掩码，排除值为-9999的元素
    mask = (y_image_array_all != -9999)
    # 计算四分位数和IQR(四分位距)
    q1 = np.percentile(y_image_array_all[mask], 25)
    q3 = np.percentile(y_image_array_all[mask], 75)
    iqr = q3 - q1
    # 定义上下限
    # lower_bound = q1 - 3 * iqr
    # upper_bound = q3 + 3 * iqr
    # 1.5倍的四分位距
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound,upper_bound

def full_filledvalue(interpolated_array):
    invalid_indices = np.any(interpolated_array == -9999, axis=2)
    reshaped_indices = np.repeat(invalid_indices[:, :, np.newaxis], interpolated_array.shape[2], axis=2)
    interpolated_array[reshaped_indices] = -9999
    return interpolated_array

# 用于判断x和y的插值与否，y的质量控制
def mul_match_x_y_modis_quality(x_image_array_all0, y_image_array_all0,x_inter_mask00, y_qa_array_all0,ref=True, lai=True, meteorology=True, era5_meteorology = True,NEE_GPP_RECO=True,
                        ref_quality=0, ref_filledvalue=0, lai_filledvalue=0, lai_quality=0, era5_mete_filledvalue =0,meteorology_filledvalue=0,
                        NEE_GPP_RECO_filledvalue=0,is_inter = 0):
    # 根据y值的范围做进一步筛选，根据阈值范围或者
    NEE_lower_bound, NEE_upper_bound = interquartile_range(y_image_array_all0[:, :, 6])
    GPP_lower_bound, GPP_upper_bound = interquartile_range(y_image_array_all0[:, :, 7])
    RECO_lower_bound, RECO_upper_bound = interquartile_range(y_image_array_all0[:, :, 8])
    # 创建掩码
    # **Step 1: 创建 mask1（第一个条件）**
    mask1 = np.any((x_inter_mask00 == -9999) | (x_inter_mask00 == 1), axis=(2, 3, 4))  # 形状 (1688, 365)
    # mask1 = np.all((x_inter_mask00[:, :, 0, :, :] == -9999) | (x_inter_mask00[:, :, 0, :, :] == 1), axis=(2, 3))  # 形状 (1688, 365)
    # mask1 = np.any((x_inter_mask00 == -9999))  # 形状 (1688, 365)
    # **Step 2: 创建 mask2（第一个条件 + 第二个条件）**
    mask2 = (y_qa_array_all0[:, :, 3] < 0) | (y_qa_array_all0[:, :, 3] > 0.2)  # 形状 (1688, 365)

    mask3 = (y_image_array_all0[:, :, 6] == -9999)
    # **Step 3: 根据 is_inter 选择掩码**
    # is_inter = 2  # 假设 is_inter 为 1 或 2
    if is_inter == 0:
        # 保持原始数组
        x_image_array_all = x_image_array_all0
        y_image_array_all = y_image_array_all0
        x_inter_mask = x_inter_mask00
        y_qa_array_all = y_qa_array_all0
    elif is_inter == 1:
        # **Step 3: 仅使用 mask1 过滤 x_image_array**
        combined_mask = mask1 # 形状 (1688, 365)
        x_image_array_all = x_image_array_all0.copy()
        x_image_array_all[combined_mask, :, :, :] = -9999 # 仅第一个条件
        y_image_array_all = y_image_array_all0
        x_inter_mask = x_inter_mask00
        y_qa_array_all = y_qa_array_all0
    elif is_inter == 2:
        combined_mask = np.logical_or(mask1, mask2)
        # **Step 5: 使用组合 mask 过滤 x_image_array 和 y_image_array**
        # combined_mask = np.logical_or.reduce([mask1, mask2,mask3])
        x_image_array_all  = x_image_array_all0.copy()
        y_image_array_all = y_image_array_all0.copy()
        x_image_array_all[combined_mask, :, :, :] = -9999
        y_image_array_all[combined_mask, :] = -9999  # 仅对 y_image_array 进行处理
        x_inter_mask = x_inter_mask00
        y_qa_array_all = y_qa_array_all0
    else:
        raise ValueError("is_inter 必须是 0, 1 或 2")

    x_image_array_all0[0,:,:,1,1]
    x_image_array_all[0,:,:,1,1]
    y_image_array_all0[:,:,6]
    y_image_array_all
    x_inter_mask00[:,:,0,1,1]
    y_qa_array_all
    y_qa_array_all0[:,:,3]

    y_index = []
    y_difference_index = []
    y_positive_index = []
    for i in range(y_image_array_all.shape[0]):
        for j in range(y_image_array_all.shape[1]):
            if (np.all(y_image_array_all[i, j, 8] >= RECO_upper_bound) or np.all(y_image_array_all[i, j, 8] <= RECO_lower_bound)) \
                    or (np.all(y_image_array_all[i, j, 7] >= GPP_upper_bound) or np.all(y_image_array_all[i, j, 7] <= GPP_lower_bound)) \
                    or (np.all(y_image_array_all[i, j, 6] >= NEE_upper_bound) or np.all(y_image_array_all[i, j, 6] <= NEE_lower_bound)):
                y_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
            # elif (y_image_array_all[i, j, 6] != -9999) and (y_image_array_all[i, j, 7] != -9999) and (y_image_array_all[i, j, 8] != -9999) \
            #         and ((y_image_array_all[i, j, 7] - y_image_array_all[i, j, 8] - np.abs(y_image_array_all[i, j, 6])) > 0.1):
            elif (y_image_array_all[i, j, 6] != -9999) and (y_image_array_all[i, j, 7] != -9999) and (y_image_array_all[i, j, 8] != -9999) \
                 and np.abs((y_image_array_all[i, j, 8] - y_image_array_all[i, j, 7]) - y_image_array_all[i, j, 6]) > 0.1:
                y_difference_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
            elif (y_image_array_all[i, j, 7] < 0) or (y_image_array_all[i, j, 8] < 0):
                y_positive_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
    # y_image_array_all[1775, 273, 6:9]
    ## 先找出x的有效值，然后根据索引筛选出对应的y值
    x_y_indices = []
    x1_filled = []
    x1_nonfilled = []
    x1_valid_quality_good = []
    x1_valid_quality_bad = []
    x1_quality_all = []
    x1_all = []

    x2_filled = []
    x2_nonfilled = []
    x2_valid_quality_good = []
    x2_valid_quality_bad = []
    x2_qaulity_all = []
    x2_all = []

    x3_valid = []
    x3_invalid = []
    x3_all = []
    x4_valid = []
    x4_invalid = []
    x4_all = []
    y_valid = []
    y_invalid = []
    y_all = []

    for i in range(x_image_array_all.shape[0]):
        for j in range(x_image_array_all.shape[1]):
            if ref is True:
                # 判断ref值是否为填充值,0 不等于填充值
                if ref_filledvalue == 0:
                    if np.any(x_image_array_all[i, j, :7, :, :] != -9999):
                        x1_nonfilled.append((i, j))
                elif ref_filledvalue == 1:
                    if np.all(x_image_array_all[i, j, :7, :, :] == -9999):
                        x1_filled.append((i, j))
                elif ref_filledvalue == 2:
                    x1_all.append((i, j))

                # 判断ref值的质量，0：good;1,bad,2:good and bad
                if ref_quality == 0:
                    if np.any(x_image_array_all[i, j, 7:14, :, :] != -9999) and np.any(
                            x_image_array_all[i, j, 7:14, :, :] != 1):
                        x1_valid_quality_good.append((i, j))
                elif ref_quality == 1:
                    if np.all(x_image_array_all[i, j, 7:14, :, :] == -9999) or np.all(
                            x_image_array_all[i, j, 7:14, :, :] == 1):
                        x_image_array_all[i, j, :7, :, :] = -9999
                        x1_valid_quality_bad.append((i, j))
                # else:
                #     x1_quality_all.append((i, j))
                elif ref_quality == 2:
                    x1_quality_all.append((i, j))

            if lai is True:
                # lai为质量好的有效值（qa ≠1）；不等于-9999
                if lai_filledvalue == 0:
                    if np.any(x_image_array_all[i, j, 14:16, :, :] != -9999):
                        x2_nonfilled.append((i, j))
                elif lai_filledvalue == 1:
                    if np.all(x_image_array_all[i, j, 14:16, :, :] == -9999):
                        x2_filled.append((i, j))
                else:
                    x2_all.append((i, j))

                # 判断ref值的质量，0：good;1,bad,2:good and bad
                if lai_quality == 0:
                    if np.any(x_image_array_all[i, j, 16:17, :, :] != -9999) & np.any(
                            x_image_array_all[i, j, 16:17, :, :] != 1):
                        x2_valid_quality_good.append((i, j))
                elif lai_quality == 1:
                    if np.all(x_image_array_all[i, j, 16:17, :, :] == -9999) | np.all(
                            x_image_array_all[i, j, 16:17, :, :] == 1):
                        x_image_array_all[i, j, 16:17, :, :] = -9999
                        x2_valid_quality_bad.append((i, j))
                # else:
                #     x2_qaulity_all.append((i, j))
                elif lai_quality == 2:
                    x2_qaulity_all.append((i, j))

            if meteorology is True:
                if meteorology_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(y_image_array_all[i, j, 9:13] != -9999)):
                        x3_valid.append((i, j))
                elif meteorology_filledvalue == 1:
                    if (np.all(y_image_array_all[i, j, 9:13] == -9999)):
                        x3_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    x3_all.append((i, j))

            # 针对ERA5-Land 气象数据
            if era5_meteorology is True:
                if era5_mete_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(x_image_array_all[i, j, 17:, :, :] != -9999)):
                        x4_valid.append((i, j))
                elif era5_mete_filledvalue == 1:
                    if (np.all(x_image_array_all[i, j, 17:, :, :] == -9999)):
                        x4_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    x4_all.append((i, j))

            if NEE_GPP_RECO is True:
                if NEE_GPP_RECO_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(y_image_array_all[i, j, 6:9] != -9999)):
                        y_valid.append((i, j))
                elif NEE_GPP_RECO_filledvalue == 1:
                    if (np.all(y_image_array_all[i, j, 6:9] == -9999)):
                        y_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    y_all.append((i, j))
            else:
                continue
                # x_y_indices.append((i, j))
                # continue

    # x_y_indices = x1_nonfilled + x1_filled + x1_all + x1_valid_quality_good + x1_valid_quality_bad + x1_quality_all +\
    #                         x2_nonfilled + x2_filled + x2_all + x2_valid_quality_good + x2_valid_quality_bad + x2_qaulity_all + \
    #                         x3_valid + x3_invalid + x3_all + y_valid + y_invalid + y_all
    # print(len(x_y_indices))
    all_lists = [x1_nonfilled, x1_filled, x1_all, x1_valid_quality_good, x1_valid_quality_bad, x1_quality_all,
                 x2_nonfilled, x2_filled, x2_all, x2_valid_quality_good, x2_valid_quality_bad, x2_qaulity_all,
                 x3_valid, x3_invalid, x3_all, x4_valid, x4_invalid, x4_all,y_valid, y_invalid, y_all]
    print('hhhh')
    x_image_array_all[:,:,17,1,1]
    #*****************************************
    # 使用reduce函数找出同时出现在所有非空列表中的元组
    common_tuples = reduce(lambda x, y: set(x) & set(y), filter(None, all_lists))

    print(len(common_tuples))
    x_image_array_all[:, :, 17, 1, 1]
    y_image_array_all[:, :, 0]
    sorted_x_all_y_valid_indices = sorted(common_tuples, key=lambda x: (x[0], x[1]))

    selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    inter_bands = [0]
    if lai == True:
        # 若只使用fpar
        selected_bands.extend([14, 15])  # Bands for reflectance and LAI
        # # 若使用fpar和lai
        inter_bands.extend([1])
    if era5_meteorology == True:
        # Bands for reflectance and ERA5 meteorology
        selected_bands.extend([17, 18, 19, 20, 21, 22])
        inter_bands.extend([2])
    x_image_array0 = x_image_array_all[:, :, selected_bands, :, :]
    x_mask_array0 = x_inter_mask[:, :, inter_bands, :, :]

    # 如果某个波段为填充值，则所有波段均为填充值
    x_image_array0 = full_filledvalue(x_image_array0)

    # 所有site year的数据
    x_image_array_list = []
    y_image_array_list = []
    x_inter_mask_list = []
    y_qa_list = []
    # 按照索引的第一个数进行分组，并将每个小数组追加到对应的大数组中
    grouped_indices = {}
    for index in sorted_x_all_y_valid_indices:
        if index[0] not in grouped_indices:
            grouped_indices[index[0]] = []
        grouped_indices[index[0]].append(index)

    # 根据分组的索引构建大数组
    for site_year, indices in grouped_indices.items():
        one_site_year_x_data = []
        one_site_year_y_data = []
        one_site_year_x_mask_data = []
        one_site_year_y_qa_data = []
        for index in indices:
            x_image_array_selected = x_image_array0[index]
            y_image_array_selected = y_image_array_all[index]
            x_mask_array_selected = x_mask_array0[index]
            y_qa_array_selected = y_qa_array_all[index]
            one_site_year_x_data.append(x_image_array_selected)
            one_site_year_y_data.append(y_image_array_selected)
            one_site_year_x_mask_data.append(x_mask_array_selected)
            one_site_year_y_qa_data.append(y_qa_array_selected)
        x_image_array_list.append(np.array(one_site_year_x_data))
        y_image_array_list.append(np.array(one_site_year_y_data))
        x_inter_mask_list.append(np.array(one_site_year_x_mask_data))
        y_qa_list.append(np.array(one_site_year_y_qa_data))
    new_x_array = x_image_array_list
    new_y_array = y_image_array_list
    new_x_mask_array = x_inter_mask_list
    new_y_qa_array = y_qa_list

    # 根据new_array最大的time series 进行padding
    # 获取所有列表的最大记录数
    max_records = max(len(arr) for arr in new_y_array)
    # 对每个列表进行填充
    padded_y_array = []
    for arr in new_y_array:
        padded_y_arr = np.pad(arr, ((0, max_records - len(arr)), (0, 0)), mode='constant', constant_values=-9999)
        padded_y_array.append(padded_y_arr)
    # 同理，对x_影像进行padding
    max_shape = max([arrx.shape for arrx in new_x_array])
    # 对每个数组进行填充
    padded_x_array = []
    for arrx in new_x_array:
        pad_width = [(0, max_shape[i] - arrx.shape[i]) for i in range(len(max_shape))]
        padded_arr = np.pad(arrx, pad_width, mode='constant', constant_values=-9999)
        padded_x_array.append(padded_arr)

    # 同理，对x_影像进行padding
    max_x_shape = max([arrxx.shape for arrxx in new_x_mask_array])
    # 对每个数组进行填充
    padded_x_mask_array = []
    for arrxx in new_x_mask_array:
        pad_x_width = [(0, max_x_shape[i] - arrxx.shape[i]) for i in range(len(max_x_shape))]
        padded_x_arr = np.pad(arrxx, pad_x_width, mode='constant', constant_values=-9999)
        padded_x_mask_array.append(padded_x_arr)

    # 获取所有列表的最大记录数
    max_y_qa_records = max(len(arr_y) for arr_y in new_y_qa_array)
    # 对每个列表进行填充
    padded_y_qa_array = []
    for arr_y in new_y_qa_array:
        padded_y_qa = np.pad(arr_y, ((0, max_y_qa_records - len(arr_y)), (0, 0)), mode='constant', constant_values=-9999)
        padded_y_qa_array.append(padded_y_qa)

    # 将填充后的列表转换回数组形式
    padded_y_array = np.stack(padded_y_array)
    y_qa_array = np.stack(padded_y_qa_array)
    # 删除（不要最后一列（站点名称的数字表示））
    y_image_array = padded_y_array[:, :, :]
    x_image_array = np.stack(padded_x_array)
    x_mask_array = np.stack(padded_x_mask_array)

    # selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    # inter_bands = [0]
    # if lai == True:
    #     # 若只使用fpar
    #     selected_bands.extend([14, 15])  # Bands for reflectance and LAI
    #     # # 若使用fpar和lai
    #     inter_bands.extend([1])
    # if era5_meteorology == True:
    #     # Bands for reflectance and ERA5 meteorology
    #     selected_bands.extend([17, 18, 19, 20, 21, 22])
    #     inter_bands.extend([2])
    # x_image_array = x_image_array0[:, :, selected_bands, :, :]
    # x_mask_array = x_mask_array0[:, :, inter_bands, :, :]

    # 过滤掉有y 无x的情况
    x_all_valid_indices11 = []
    x_all_invalid_indices11 = []
    # for m in range(x_image_array.shape[0]):
    #     if np.any(x_image_array[m] != -9999):
    #         x_all_valid_indices11.append(m)
    #     else:
    #         x_all_invalid_indices11.append(m)
    for m in range(x_image_array.shape[0]):
        if np.all(x_image_array[m] == -9999):
            x_all_invalid_indices11.append(m)
        else:
            x_all_valid_indices11.append(m)
    x_image_array11 = x_image_array[x_all_valid_indices11]
    y_image_array11 = y_image_array[x_all_valid_indices11]
    x_mask_array11 = x_mask_array[x_all_valid_indices11]
    y_qa_array11 = y_qa_array[x_all_valid_indices11]

    # 过滤掉所有doy的y值全为-9999的siteyear
    y_all_valid_indices22 = []
    for n in range(y_image_array11.shape[0]):
        if np.any(y_image_array11[n, :, 6:9] != -9999):
            # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
            y_all_valid_indices22.append(n)
    x_image_array22 = x_image_array11[y_all_valid_indices22]
    y_image_array22 = y_image_array11[y_all_valid_indices22]
    x_mask_array22 = x_mask_array11[y_all_valid_indices22]
    y_qa_array22 = y_qa_array11[y_all_valid_indices22]
    # 有x,也有y，但是彼此都不对应；整个site year的所有doy里面没有一个是x和y均有效的数据
    result = []
    for i in range(x_image_array22.shape[0]):
        has_valid_doy = False
        for j in range(x_image_array22.shape[1]):
            # if np.any(x_image_array22[i, j] != -9999) and np.all(y_image_array_all[i, j, 5] != -9999) and np.all(y_image_array_all[i, j, 6] != -9999)  and np.all(y_image_array_all[i, j, 7] != -9999):
            if np.any(x_image_array22[i, j] != -9999) and np.any(y_image_array22[i, j, 6:9] != -9999):
                has_valid_doy = True
                break
        if not has_valid_doy:
            result.append(i)
    mask = np.isin(np.arange(len(x_image_array22)), result, invert=True)
    x_image_array33 = x_image_array22[mask]
    y_image_array33 = y_image_array22[mask]
    x_mask_array33 = x_mask_array22[mask]
    y_qa_array33 = y_qa_array22[mask]
    # 删除一整年的x值都是无效的数据
    x_image_array33 = full_filledvalue(x_image_array33)
    # 将一整年都为-9999的site-year 删除
    x_all_invalid_indices33 = []
    for n in range(x_image_array33.shape[0]):
        if np.all(x_image_array33[n, ...] == -9999):
            # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
            x_all_invalid_indices33.append(n)
    x_image_array44 = x_image_array33[~np.in1d(np.arange(x_image_array33.shape[0]), x_all_invalid_indices33)]
    y_image_array44 = y_image_array33[~np.in1d(np.arange(y_image_array33.shape[0]), x_all_invalid_indices33)]
    x_mask_array44 = x_mask_array33[~np.in1d(np.arange(x_mask_array33.shape[0]), x_all_invalid_indices33)]
    # x_image_array44[217,:,:,2,2]
    y_qa_array44 = y_qa_array33[~np.in1d(np.arange(y_qa_array33.shape[0]), x_all_invalid_indices33)]

    # selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    # if ref == True:
    #     if lai == True:
    #         # 若只使用fpar
    #         selected_bands.extend([14,15])   # Bands for reflectance and LAI
    #         # # 若使用fpar和lai
    #         # selected_bands.extend([14,15])  # Bands for reflectance and LAI
    #     if era5_meteorology == True:
    #          # Bands for reflectance and ERA5 meteorology
    #          selected_bands.extend([17, 18, 19, 20,21, 22])
    # x_image_array_55 = x_image_array44[:, :, selected_bands, :, :]
    # 检查y_image_array_all中[:,:,5]、[:,:,6]和[:,:,7]列数据不等于-9999的个数
    # 创建布尔掩码，标识不等于-9999的元素
    mask_5 = (y_image_array44[:, :, 6] != -9999)
    mask_6 = (y_image_array44[:, :, 7] != -9999)
    mask_7 = (y_image_array44[:, :, 8] != -9999)
    # 计算不等于-9999的元素个数
    count_5 = np.sum(mask_5)
    count_6 = np.sum(mask_6)
    count_7 = np.sum(mask_7)
    print("NEE数据不等于-9999的个数：", count_5)
    print("GPP数据不等于-9999的个数：", count_6)
    print("RECO数据不等于-9999的个数：", count_7)
    # 删除23个特征里面的第2个特征（即 第2维度的第一个index,为了和之前的数组保持一致，且删掉的是经度，倒数第2个也是，所以不影响）
    # y_image_array_44 = np.delete(y_image_array33, 1, axis=2)
    y_image_array_55 = y_image_array44
    x_mask_array55 = x_mask_array44
    y_qa_array_55 = y_qa_array44
    x_image_array_55 = x_image_array44
    return x_image_array_55, y_image_array_55,x_mask_array55,y_qa_array_55

def mul_match_x_y_modis(x_image_array_all, y_image_array_all,x_inter_mask, y_qa_array_all,ref=True, lai=True, meteorology=True, era5_meteorology = True,NEE_GPP_RECO=True,
                        ref_quality=0, ref_filledvalue=0, lai_filledvalue=0, lai_quality=0, era5_mete_filledvalue =0,meteorology_filledvalue=0,
                        NEE_GPP_RECO_filledvalue=0):
    # 根据y值的范围做进一步筛选，根据阈值范围或者
    NEE_lower_bound, NEE_upper_bound = interquartile_range(y_image_array_all[:, :, 6])
    GPP_lower_bound, GPP_upper_bound = interquartile_range(y_image_array_all[:, :, 7])
    RECO_lower_bound, RECO_upper_bound = interquartile_range(y_image_array_all[:, :, 8])
    y_index = []
    y_difference_index = []
    y_positive_index = []
    for i in range(y_image_array_all.shape[0]):
        for j in range(y_image_array_all.shape[1]):
            if (np.all(y_image_array_all[i, j, 8] >= RECO_upper_bound) or np.all(y_image_array_all[i, j, 8] <= RECO_lower_bound)) \
                    or (np.all(y_image_array_all[i, j, 7] >= GPP_upper_bound) or np.all(y_image_array_all[i, j, 7] <= GPP_lower_bound)) \
                    or (np.all(y_image_array_all[i, j, 6] >= NEE_upper_bound) or np.all(y_image_array_all[i, j, 6] <= NEE_lower_bound)):
                y_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
            # elif (y_image_array_all[i, j, 6] != -9999) and (y_image_array_all[i, j, 7] != -9999) and (y_image_array_all[i, j, 8] != -9999) \
            #         and ((y_image_array_all[i, j, 7] - y_image_array_all[i, j, 8] - np.abs(y_image_array_all[i, j, 6])) > 0.1):
            elif (y_image_array_all[i, j, 6] != -9999) and (y_image_array_all[i, j, 7] != -9999) and (y_image_array_all[i, j, 8] != -9999) \
                 and np.abs((y_image_array_all[i, j, 8] - y_image_array_all[i, j, 7]) - y_image_array_all[i, j, 6]) > 0.1:
                y_difference_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
            elif (y_image_array_all[i, j, 7] < 0) or (y_image_array_all[i, j, 8] < 0):
                y_positive_index.append((i, j))
                y_image_array_all[i, j, 6:9] = -9999
    # y_image_array_all[1775, 273, 6:9]
    ## 先找出x的有效值，然后根据索引筛选出对应的y值
    x_y_indices = []
    x1_filled = []
    x1_nonfilled = []
    x1_valid_quality_good = []
    x1_valid_quality_bad = []
    x1_quality_all = []
    x1_all = []

    x2_filled = []
    x2_nonfilled = []
    x2_valid_quality_good = []
    x2_valid_quality_bad = []
    x2_qaulity_all = []
    x2_all = []

    x3_valid = []
    x3_invalid = []
    x3_all = []
    x4_valid = []
    x4_invalid = []
    x4_all = []
    y_valid = []
    y_invalid = []
    y_all = []

    for i in range(x_image_array_all.shape[0]):
        for j in range(x_image_array_all.shape[1]):
            if ref is True:
                # 判断ref值是否为填充值,0 不等于填充值
                if ref_filledvalue == 0:
                    if np.any(x_image_array_all[i, j, :7, :, :] != -9999):
                        x1_nonfilled.append((i, j))
                elif ref_filledvalue == 1:
                    if np.all(x_image_array_all[i, j, :7, :, :] == -9999):
                        x1_filled.append((i, j))
                elif ref_filledvalue == 2:
                    x1_all.append((i, j))

                # 判断ref值的质量，0：good;1,bad,2:good and bad
                if ref_quality == 0:
                    if np.any(x_image_array_all[i, j, 7:14, :, :] != -9999) and np.any(
                            x_image_array_all[i, j, 7:14, :, :] != 1):
                        x1_valid_quality_good.append((i, j))
                elif ref_quality == 1:
                    if np.all(x_image_array_all[i, j, 7:14, :, :] == -9999) or np.all(
                            x_image_array_all[i, j, 7:14, :, :] == 1):
                        x_image_array_all[i, j, :7, :, :] = -9999
                        x1_valid_quality_bad.append((i, j))
                # else:
                #     x1_quality_all.append((i, j))
                elif ref_quality == 2:
                    x1_quality_all.append((i, j))

            if lai is True:
                # lai为质量好的有效值（qa ≠1）；不等于-9999
                if lai_filledvalue == 0:
                    if np.any(x_image_array_all[i, j, 14:16, :, :] != -9999):
                        x2_nonfilled.append((i, j))
                elif lai_filledvalue == 1:
                    if np.all(x_image_array_all[i, j, 14:16, :, :] == -9999):
                        x2_filled.append((i, j))
                else:
                    x2_all.append((i, j))

                # 判断ref值的质量，0：good;1,bad,2:good and bad
                if lai_quality == 0:
                    if np.any(x_image_array_all[i, j, 16:17, :, :] != -9999) & np.any(
                            x_image_array_all[i, j, 16:17, :, :] != 1):
                        x2_valid_quality_good.append((i, j))
                elif lai_quality == 1:
                    if np.all(x_image_array_all[i, j, 16:17, :, :] == -9999) | np.all(
                            x_image_array_all[i, j, 16:17, :, :] == 1):
                        x_image_array_all[i, j, 16:17, :, :] = -9999
                        x2_valid_quality_bad.append((i, j))
                # else:
                #     x2_qaulity_all.append((i, j))
                elif lai_quality == 2:
                    x2_qaulity_all.append((i, j))

            if meteorology is True:
                if meteorology_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(y_image_array_all[i, j, 9:13] != -9999)):
                        x3_valid.append((i, j))
                elif meteorology_filledvalue == 1:
                    if (np.all(y_image_array_all[i, j, 9:13] == -9999)):
                        x3_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    x3_all.append((i, j))

            # 针对ERA5-Land 气象数据
            if era5_meteorology is True:
                if era5_mete_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(x_image_array_all[i, j, 17:, :, :] != -9999)):
                        x4_valid.append((i, j))
                elif era5_mete_filledvalue == 1:
                    if (np.all(x_image_array_all[i, j, 17:, :, :] == -9999)):
                        x4_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    x4_all.append((i, j))

            if NEE_GPP_RECO is True:
                if NEE_GPP_RECO_filledvalue == 0:
                    # meteorology为有效值(不等于-9999)
                    if (np.any(y_image_array_all[i, j, 6:9] != -9999)):
                        y_valid.append((i, j))
                elif NEE_GPP_RECO_filledvalue == 1:
                    if (np.all(y_image_array_all[i, j, 6:9] == -9999)):
                        y_invalid.append((i, j))
                # lai既包含有效值又包含无效值(=-9999)
                else:
                    y_all.append((i, j))
            else:
                continue
                # x_y_indices.append((i, j))
                # continue

    # x_y_indices = x1_nonfilled + x1_filled + x1_all + x1_valid_quality_good + x1_valid_quality_bad + x1_quality_all +\
    #                         x2_nonfilled + x2_filled + x2_all + x2_valid_quality_good + x2_valid_quality_bad + x2_qaulity_all + \
    #                         x3_valid + x3_invalid + x3_all + y_valid + y_invalid + y_all
    # print(len(x_y_indices))
    all_lists = [x1_nonfilled, x1_filled, x1_all, x1_valid_quality_good, x1_valid_quality_bad, x1_quality_all,
                 x2_nonfilled, x2_filled, x2_all, x2_valid_quality_good, x2_valid_quality_bad, x2_qaulity_all,
                 x3_valid, x3_invalid, x3_all, x4_valid, x4_invalid, x4_all,y_valid, y_invalid, y_all]
    print('hhhh')
    x_image_array_all[:,:,17,1,1]
    #*****************************************
    # 使用reduce函数找出同时出现在所有非空列表中的元组
    common_tuples = reduce(lambda x, y: set(x) & set(y), filter(None, all_lists))
    print(len(common_tuples))
    x_image_array_all[:, :, 17, 1, 1]
    y_image_array_all[:, :, 0]
    sorted_x_all_y_valid_indices = sorted(common_tuples, key=lambda x: (x[0], x[1]))

    selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    inter_bands = [0]
    if lai == True:
        # 若只使用fpar
        selected_bands.extend([14, 15])  # Bands for reflectance and LAI
        # # 若使用fpar和lai
        inter_bands.extend([1])
    if era5_meteorology == True:
        # Bands for reflectance and ERA5 meteorology
        selected_bands.extend([17, 18, 19, 20, 21, 22])
        inter_bands.extend([2])
    x_image_array0 = x_image_array_all[:, :, selected_bands, :, :]
    x_mask_array0 = x_inter_mask[:, :, inter_bands, :, :]

    # 如果某个波段为填充值，则所有波段均为填充值
    x_image_array0 = full_filledvalue(x_image_array0)

    # 所有site year的数据
    x_image_array_list = []
    y_image_array_list = []
    x_inter_mask_list = []
    y_qa_list = []
    # 按照索引的第一个数进行分组，并将每个小数组追加到对应的大数组中
    grouped_indices = {}
    for index in sorted_x_all_y_valid_indices:
        if index[0] not in grouped_indices:
            grouped_indices[index[0]] = []
        grouped_indices[index[0]].append(index)

    # 根据分组的索引构建大数组
    for site_year, indices in grouped_indices.items():
        one_site_year_x_data = []
        one_site_year_y_data = []
        one_site_year_x_mask_data = []
        one_site_year_y_qa_data = []
        for index in indices:
            x_image_array_selected = x_image_array0[index]
            y_image_array_selected = y_image_array_all[index]
            x_mask_array_selected = x_mask_array0[index]
            y_qa_array_selected = y_qa_array_all[index]
            one_site_year_x_data.append(x_image_array_selected)
            one_site_year_y_data.append(y_image_array_selected)
            one_site_year_x_mask_data.append(x_mask_array_selected)
            one_site_year_y_qa_data.append(y_qa_array_selected)
        x_image_array_list.append(np.array(one_site_year_x_data))
        y_image_array_list.append(np.array(one_site_year_y_data))
        x_inter_mask_list.append(np.array(one_site_year_x_mask_data))
        y_qa_list.append(np.array(one_site_year_y_qa_data))
    new_x_array = x_image_array_list
    new_y_array = y_image_array_list
    new_x_mask_array = x_inter_mask_list
    new_y_qa_array = y_qa_list

    # 根据new_array最大的time series 进行padding
    # 获取所有列表的最大记录数
    max_records = max(len(arr) for arr in new_y_array)
    # 对每个列表进行填充
    padded_y_array = []
    for arr in new_y_array:
        padded_y_arr = np.pad(arr, ((0, max_records - len(arr)), (0, 0)), mode='constant', constant_values=-9999)
        padded_y_array.append(padded_y_arr)
    # 同理，对x_影像进行padding
    max_shape = max([arrx.shape for arrx in new_x_array])
    # 对每个数组进行填充
    padded_x_array = []
    for arrx in new_x_array:
        pad_width = [(0, max_shape[i] - arrx.shape[i]) for i in range(len(max_shape))]
        padded_arr = np.pad(arrx, pad_width, mode='constant', constant_values=-9999)
        padded_x_array.append(padded_arr)

    # 同理，对x_影像进行padding
    max_x_shape = max([arrxx.shape for arrxx in new_x_mask_array])
    # 对每个数组进行填充
    padded_x_mask_array = []
    for arrxx in new_x_mask_array:
        pad_x_width = [(0, max_x_shape[i] - arrxx.shape[i]) for i in range(len(max_x_shape))]
        padded_x_arr = np.pad(arrxx, pad_x_width, mode='constant', constant_values=-9999)
        padded_x_mask_array.append(padded_x_arr)

    # 获取所有列表的最大记录数
    max_y_qa_records = max(len(arr_y) for arr_y in new_y_qa_array)
    # 对每个列表进行填充
    padded_y_qa_array = []
    for arr_y in new_y_qa_array:
        padded_y_qa = np.pad(arr_y, ((0, max_y_qa_records - len(arr_y)), (0, 0)), mode='constant', constant_values=-9999)
        padded_y_qa_array.append(padded_y_qa)

    # 将填充后的列表转换回数组形式
    padded_y_array = np.stack(padded_y_array)
    y_qa_array = np.stack(padded_y_qa_array)
    # 删除（不要最后一列（站点名称的数字表示））
    y_image_array = padded_y_array[:, :, :]
    x_image_array = np.stack(padded_x_array)
    x_mask_array = np.stack(padded_x_mask_array)

    # selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    # inter_bands = [0]
    # if lai == True:
    #     # 若只使用fpar
    #     selected_bands.extend([14, 15])  # Bands for reflectance and LAI
    #     # # 若使用fpar和lai
    #     inter_bands.extend([1])
    # if era5_meteorology == True:
    #     # Bands for reflectance and ERA5 meteorology
    #     selected_bands.extend([17, 18, 19, 20, 21, 22])
    #     inter_bands.extend([2])
    # x_image_array = x_image_array0[:, :, selected_bands, :, :]
    # x_mask_array = x_mask_array0[:, :, inter_bands, :, :]

    # 过滤掉有y 无x的情况
    x_all_valid_indices11 = []
    x_all_invalid_indices11 = []
    # for m in range(x_image_array.shape[0]):
    #     if np.any(x_image_array[m] != -9999):
    #         x_all_valid_indices11.append(m)
    #     else:
    #         x_all_invalid_indices11.append(m)
    for m in range(x_image_array.shape[0]):
        if np.all(x_image_array[m] == -9999):
            x_all_invalid_indices11.append(m)
        else:
            x_all_valid_indices11.append(m)
    x_image_array11 = x_image_array[x_all_valid_indices11]
    y_image_array11 = y_image_array[x_all_valid_indices11]
    x_mask_array11 = x_mask_array[x_all_valid_indices11]
    y_qa_array11 = y_qa_array[x_all_valid_indices11]

    # 过滤掉所有doy的y值全为-9999的siteyear
    y_all_valid_indices22 = []
    for n in range(y_image_array11.shape[0]):
        if np.any(y_image_array11[n, :, 6:9] != -9999):
            # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
            y_all_valid_indices22.append(n)
    x_image_array22 = x_image_array11[y_all_valid_indices22]
    y_image_array22 = y_image_array11[y_all_valid_indices22]
    x_mask_array22 = x_mask_array11[y_all_valid_indices22]
    y_qa_array22 = y_qa_array11[y_all_valid_indices22]
    # 有x,也有y，但是彼此都不对应；整个site year的所有doy里面没有一个是x和y均有效的数据
    result = []
    for i in range(x_image_array22.shape[0]):
        has_valid_doy = False
        for j in range(x_image_array22.shape[1]):
            # if np.any(x_image_array22[i, j] != -9999) and np.all(y_image_array_all[i, j, 5] != -9999) and np.all(y_image_array_all[i, j, 6] != -9999)  and np.all(y_image_array_all[i, j, 7] != -9999):
            if np.any(x_image_array22[i, j] != -9999) and np.any(y_image_array22[i, j, 6:9] != -9999):
                has_valid_doy = True
                break
        if not has_valid_doy:
            result.append(i)
    mask = np.isin(np.arange(len(x_image_array22)), result, invert=True)
    x_image_array33 = x_image_array22[mask]
    y_image_array33 = y_image_array22[mask]
    x_mask_array33 = x_mask_array22[mask]
    y_qa_array33 = y_qa_array22[mask]
    # 删除一整年的x值都是无效的数据
    x_image_array33 = full_filledvalue(x_image_array33)
    # 将一整年都为-9999的site-year 删除
    x_all_invalid_indices33 = []
    for n in range(x_image_array33.shape[0]):
        if np.all(x_image_array33[n, ...] == -9999):
            # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
            x_all_invalid_indices33.append(n)
    x_image_array44 = x_image_array33[~np.in1d(np.arange(x_image_array33.shape[0]), x_all_invalid_indices33)]
    y_image_array44 = y_image_array33[~np.in1d(np.arange(y_image_array33.shape[0]), x_all_invalid_indices33)]
    x_mask_array44 = x_mask_array33[~np.in1d(np.arange(x_mask_array33.shape[0]), x_all_invalid_indices33)]
    x_image_array44[217,:,:,2,2]
    y_qa_array44 = y_qa_array33[~np.in1d(np.arange(y_qa_array33.shape[0]), x_all_invalid_indices33)]

    # selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    # if ref == True:
    #     if lai == True:
    #         # 若只使用fpar
    #         selected_bands.extend([14,15])   # Bands for reflectance and LAI
    #         # # 若使用fpar和lai
    #         # selected_bands.extend([14,15])  # Bands for reflectance and LAI
    #     if era5_meteorology == True:
    #          # Bands for reflectance and ERA5 meteorology
    #          selected_bands.extend([17, 18, 19, 20,21, 22])
    # x_image_array_55 = x_image_array44[:, :, selected_bands, :, :]
    # 检查y_image_array_all中[:,:,5]、[:,:,6]和[:,:,7]列数据不等于-9999的个数
    # 创建布尔掩码，标识不等于-9999的元素
    mask_5 = (y_image_array44[:, :, 6] != -9999)
    mask_6 = (y_image_array44[:, :, 7] != -9999)
    mask_7 = (y_image_array44[:, :, 8] != -9999)
    # 计算不等于-9999的元素个数
    count_5 = np.sum(mask_5)
    count_6 = np.sum(mask_6)
    count_7 = np.sum(mask_7)
    print("NEE数据不等于-9999的个数：", count_5)
    print("GPP数据不等于-9999的个数：", count_6)
    print("RECO数据不等于-9999的个数：", count_7)
    # 删除23个特征里面的第2个特征（即 第2维度的第一个index,为了和之前的数组保持一致，且删掉的是经度，倒数第2个也是，所以不影响）
    # y_image_array_44 = np.delete(y_image_array33, 1, axis=2)
    y_image_array_55 = y_image_array44
    x_mask_array55 = x_mask_array44
    y_qa_array_55 = y_qa_array44
    x_image_array_55 = x_image_array44
    return x_image_array_55, y_image_array_55,x_mask_array55,y_qa_array_55

# def mul_match_x_y_modis(x_image_array_all, y_image_array_all,ref = True,lai = True,meteorology = True,era5_meteorology = True,NEE_GPP_RECO = True,
#                         ref_quality = 0, ref_filledvalue = 0,lai_filledvalue = 0,lai_quality = 0,era5_mete_filledvalue =0, meteorology_filledvalue = 0,
#                         NEE_GPP_RECO_filledvalue = 0):
#     # 根据y值的范围做进一步筛选，根据阈值范围或者
#     NEE_lower_bound, NEE_upper_bound = interquartile_range(y_image_array_all[:, :, 6])
#     GPP_lower_bound, GPP_upper_bound = interquartile_range(y_image_array_all[:, :, 7])
#     RECO_lower_bound, RECO_upper_bound = interquartile_range(y_image_array_all[:, :, 8])
#     y_index = []
#     for i in range(y_image_array_all.shape[0]):
#         for j in range(y_image_array_all.shape[1]):
#             # NEE GPP RECO的值低于四分位数的3倍
#             if ((np.all(y_image_array_all[i, j, 8] >= RECO_upper_bound) or np.all(
#                     y_image_array_all[i, j, 8] <= RECO_lower_bound))) \
#                     or ((np.all(y_image_array_all[i, j, 7] >= GPP_upper_bound) or np.all(
#                 y_image_array_all[i, j, 7] <= GPP_lower_bound))) or \
#                     ((np.all(y_image_array_all[i, j, 6] >= NEE_upper_bound) or np.all(
#                         y_image_array_all[i, j, 6] <= NEE_lower_bound))):
#                 y_index.append((i, j))
#                 y_image_array_all[i, j, 6:9] = -9999
#             # GPP-RECO与NEE绝对值大于0.1 gC m - 2 d - 1;这里需要排除等于-9999的情况
#             elif ((y_image_array_all[i, j, 6] != -9999) & (y_image_array_all[i, j, 7] != -9999) & (y_image_array_all[i, j, 8] != -9999)) and \
#                     ((y_image_array_all[i,j,7] - y_image_array_all[i,j,8])-np.abs(y_image_array_all[i,j,6])> 0.1):
#                 y_index.append((i, j))
#                 y_image_array_all[i,j,6:9] = -9999
#             # GPP与RECO为负值
#             elif (y_image_array_all[i, j, 8] < 0) or (y_image_array_all[i, j, 9] < 0):
#                 y_index.append((i, j))
#                 y_image_array_all[i, j, 6:9] = -9999
#             else:
#                 continue
#     ## 先找出x的有效值，然后根据索引筛选出对应的y值
#     x_y_indices = []
#     x1_filled = []
#     x1_nonfilled = []
#     x1_valid_quality_good = []
#     x1_valid_quality_bad = []
#     x1_quality_all = []
#     x1_all = []
#
#     x2_filled = []
#     x2_nonfilled = []
#     x2_valid_quality_good = []
#     x2_valid_quality_bad = []
#     x2_qaulity_all = []
#     x2_all = []
#
#     x3_valid = []
#     x3_invalid = []
#     x3_all = []
#     y_valid = []
#     y_invalid = []
#     y_all = []
#
#     x4_valid = []
#     x4_invalid = []
#     x4_all = []
#
#
#     for i in range(x_image_array_all.shape[0]):
#         for j in range(x_image_array_all.shape[1]):
#             if ref is True:
#                 # 判断ref值是否为填充值,0 不等于填充值
#                 if ref_filledvalue == 0:
#                     if np.any(x_image_array_all[i, j, :7, :, :] != -9999):
#                         x1_nonfilled.append((i, j))
#                 elif ref_filledvalue == 1:
#                     if np.all(x_image_array_all[i, j, :7, :, :] == -9999):
#                         x1_filled.append((i, j))
#                 else:
#                     x1_all.append((i, j))
#
#                 # 判断ref值的质量，0：good;1,bad,2:good and bad
#                 if ref_quality == 0:
#                     if np.any(x_image_array_all[i, j, 7:14, :, :] != -9999) and np.any(
#                             x_image_array_all[i, j, 7:14, :, :] != 1):
#                         x1_valid_quality_good.append((i, j))
#                 elif ref_quality == 1:
#                     if np.all((x_image_array_all[i, j, 7:14, :, :] != -9999) or (
#                             x_image_array_all[i, j, 7:14, :, :] == 0)):
#                         x1_valid_quality_bad.append((i, j))
#                 else:
#                     x1_quality_all.append((i, j))
#
#             if lai is True:
#                 # lai为质量好的有效值（qa ≠1）；不等于-9999
#                 if lai_filledvalue == 0:
#                     if np.any(x_image_array_all[i, j, 13:14, :, :] != -9999):
#                         x2_nonfilled.append((i, j))
#                 elif ref_filledvalue == 1:
#                     if np.all(x_image_array_all[i, j, 13:14, :, :] == -9999):
#                         x2_filled.append((i, j))
#                 else:
#                     x2_all.append((i, j))
#
#                 # 判断ref值的质量，0：good;1,bad,2:good and bad
#                 if lai_quality == 0:
#                     if np.any(x_image_array_all[i, j, 15:, :, :] != -9999) & np.any(
#                             x_image_array_all[i, j, 15:, :, :] != 1):
#                         x2_valid_quality_good.append((i, j))
#                 elif ref_quality == 1:
#                     if np.all(((x_image_array_all[i, j, 15:, :, :] == -9999) | (
#                             x_image_array_all[i, j, 15:, :, :] == 1))):
#                         x2_valid_quality_bad.append((i, j))
#                 else:
#                     x2_qaulity_all.append((i, j))
#
#             # 针对flux 气象数据
#             if meteorology is True:
#                 if meteorology_filledvalue == 0:
#                     # meteorology为有效值(不等于-9999)
#                     if (np.any(y_image_array_all[i, j, 9:13] != -9999)):
#                         x3_valid.append((i, j))
#                 elif meteorology_filledvalue == 1:
#                     if (np.all(y_image_array_all[i, j, 9:13] == -9999)):
#                         x3_invalid.append((i, j))
#                 # lai既包含有效值又包含无效值(=-9999)
#                 else:
#                     x3_all.append((i, j))
#
#             # 针对ERA5-Land 气象数据
#             if era5_meteorology is True:
#                 if era5_mete_filledvalue == 0:
#                     # meteorology为有效值(不等于-9999)
#                     if (np.any(x_image_array_all[i, j, 17:, :, :] != -9999)):
#                         x4_valid.append((i, j))
#                 elif era5_mete_filledvalue == 1:
#                     if (np.all(x_image_array_all[i, j, 17:, :, :] == -9999)):
#                         x4_invalid.append((i, j))
#                 # lai既包含有效值又包含无效值(=-9999)
#                 else:
#                     x4_all.append((i, j))
#
#             if NEE_GPP_RECO is True:
#                 if NEE_GPP_RECO_filledvalue == 0:
#                     # meteorology为有效值(不等于-9999)
#                     if (np.any(y_image_array_all[i, j, 6:9] != -9999)):
#                         y_valid.append((i, j))
#                 elif NEE_GPP_RECO_filledvalue == 1:
#                     if (np.all(y_image_array_all[i, j, 6:9] == -9999)):
#                         y_invalid.append((i, j))
#                 # lai既包含有效值又包含无效值(=-9999)
#                 else:
#                     y_all.append((i, j))
#             else:
#                 continue
#                 # x_y_indices.append((i, j))
#                 # continue
#     # x_y_indices = x1_nonfilled + x1_filled + x1_all + x1_valid_quality_good + x1_valid_quality_bad + x1_quality_all +\
#     #                         x2_nonfilled + x2_filled + x2_all + x2_valid_quality_good + x2_valid_quality_bad + x2_qaulity_all + \
#     #                         x3_valid + x3_invalid + x3_all + y_valid + y_invalid + y_all
#     # print(len(x_y_indices))
#     # 筛选出同时出现在上述非空list里的元组
#     # 使用reduce函数找出同时出现在所有列表中的元组
#     # 将所有二元组列表放入一个列表中
#     all_lists = [x1_nonfilled, x1_filled, x1_all, x1_valid_quality_good, x1_valid_quality_bad, x1_quality_all,
#                  x2_nonfilled, x2_filled, x2_all, x2_valid_quality_good, x2_valid_quality_bad, x2_qaulity_all,
#                  x3_valid, x3_invalid, x3_all, x4_valid, x4_invalid, x4_all,y_valid, y_invalid, y_all]
#
#     # 使用reduce函数找出同时出现在所有非空列表中的元组
#     common_tuples = reduce(lambda x, y: set(x) & set(y), filter(None, all_lists))
#     print(len(common_tuples))
#     x_image_array_all[37, 296, 7:14, :, :]
#     y_image_array_all[0, 0, 6:9]
#     sorted_x_all_y_valid_indices = sorted(common_tuples, key=lambda x: (x[0], x[1]))
#     # 所有site year的数据
#     x_image_array_list = []
#     y_image_array_list = []
#     # 按照索引的第一个数进行分组，并将每个小数组追加到对应的大数组中
#     grouped_indices = {}
#     for index in sorted_x_all_y_valid_indices:
#         if index[0] not in grouped_indices:
#             grouped_indices[index[0]] = []
#         grouped_indices[index[0]].append(index)
#
#     # 根据分组的索引构建大数组
#     for site_year, indices in grouped_indices.items():
#         one_site_year_x_data = []
#         one_site_year_y_data = []
#         for index in indices:
#             x_image_array_selected = x_image_array_all[index]
#             y_image_array_selected = y_image_array_all[index]
#             one_site_year_x_data.append(x_image_array_selected)
#             one_site_year_y_data.append(y_image_array_selected)
#         x_image_array_list.append(np.array(one_site_year_x_data))
#         y_image_array_list.append(np.array(one_site_year_y_data))
#     new_x_array = x_image_array_list
#     new_y_array = y_image_array_list
#
#     # 根据new_array最大的time series 进行padding
#     # 获取所有列表的最大记录数
#     max_records = max(len(arr) for arr in new_y_array)
#     # 对每个列表进行填充
#     padded_y_array = []
#     for arr in new_y_array:
#         padded_y_arr = np.pad(arr, ((0, max_records - len(arr)), (0, 0)), mode='constant', constant_values=-9999)
#         padded_y_array.append(padded_y_arr)
#     # 同理，对x_影像进行padding
#     max_shape = max([arrx.shape for arrx in new_x_array])
#     # 对每个数组进行填充
#     padded_x_array = []
#     for arrx in new_x_array:
#         pad_width = [(0, max_shape[i] - arrx.shape[i]) for i in range(len(max_shape))]
#         padded_arr = np.pad(arrx, pad_width, mode='constant', constant_values=-9999)
#         padded_x_array.append(padded_arr)
#     # 将填充后的列表转换回数组形式
#     padded_y_array = np.stack(padded_y_array)
#     # 删除（不要最后一列（站点名称的数字表示））
#     y_image_array = padded_y_array[:, :, :]
#     x_image_array = np.stack(padded_x_array)
#     # 过滤掉所有doy的y值全为-9999的siteyear
#     x_all_valid_indices11 = []
#     x_all_invalid_indices11 = []
#     for m in range(x_image_array.shape[0]):
#         if np.any(x_image_array[m] != -9999):
#             x_all_valid_indices11.append(m)
#         else:
#             x_all_invalid_indices11.append(m)
#     x_image_array11 = x_image_array[x_all_valid_indices11]
#     y_image_array11 = y_image_array[x_all_valid_indices11]
#     # 过滤掉所有doy的y值全为-9999的siteyear
#     y_all_valid_indices22 = []
#     for n in range(y_image_array11.shape[0]):
#         if np.any(y_image_array11[n, :, 6:9] != -9999):
#             # if np.all(y_image_array_all[i, j, 5] != -9999) or np.all(y_image_array_all[i, j, 6] != -9999) or np.all(y_image_array_all[i, j, 7] != -9999):
#             y_all_valid_indices22.append(n)
#     x_image_array22 = x_image_array11[y_all_valid_indices22]
#     y_image_array22 = y_image_array11[y_all_valid_indices22]
#     # 有x,也有y，但是彼此都不对应；整个site year的所有doy里面没有一个是x和y均有效的数据
#     result = []
#     for i in range(x_image_array22.shape[0]):
#         has_valid_doy = False
#         for j in range(x_image_array22.shape[1]):
#             # if np.any(x_image_array22[i, j] != -9999) and np.all(y_image_array_all[i, j, 5] != -9999) and np.all(y_image_array_all[i, j, 6] != -9999)  and np.all(y_image_array_all[i, j, 7] != -9999):
#             if np.any(x_image_array22[i, j] != -9999) and np.any(y_image_array22[i, j, 6:9] != -9999):
#                 has_valid_doy = True
#                 break
#         if not has_valid_doy:
#             result.append(i)
#     mask = np.isin(np.arange(len(x_image_array22)), result, invert=True)
#     x_image_array33 = x_image_array22[mask]
#     y_image_array33 = y_image_array22[mask]
#     # 检查y_image_array_all中[:,:,5]、[:,:,6]和[:,:,7]列数据不等于-9999的个数
#     # 创建布尔掩码，标识不等于-9999的元素
#     mask_5 = (y_image_array33[:, :, 6] != -9999)
#     mask_6 = (y_image_array33[:, :, 7] != -9999)
#     mask_7 = (y_image_array33[:, :, 8] != -9999)
#     # 计算不等于-9999的元素个数
#     count_5 = np.sum(mask_5)
#     count_6 = np.sum(mask_6)
#     count_7 = np.sum(mask_7)
#     print("NEE数据不等于-9999的个数：", count_5)
#     print("GPP数据不等于-9999的个数：", count_6)
#     print("RECO数据不等于-9999的个数：", count_7)
#     return x_image_array33, y_image_array33