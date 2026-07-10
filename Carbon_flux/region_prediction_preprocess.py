import numpy as np
import os
from functools import reduce
import pandas as pd

def pre_construct_composite_norm(pre_x, pre_covariate,mean_array,stdDev_array,is_single_norm=True):
    # 打开bands_mean_st
    # 复制数据
    input_images_pre_norm0 = pre_x.copy()
    input_covariate_pre_norm0 = pre_covariate.copy()
    # ！！！注意对于-9999值的处理：值为-9999的不应该参与均值和标准差的计算
    # 但由于前面已经将空值、为云 雪等的值设为了-9999，所以不用额外设置，后续处理需要尤其注意
    # 引入屏蔽数组，屏蔽无效或者不完整的数据
    # 将测试集与训练集合并到一起，对应的均值和标准差即为所有的数据求得的
    a = np.ma.array(pre_x[:, :, :, :, :], mask=pre_x[:, :, :, :, :] == -9999.00000)
    b = np.ma.array(pre_covariate[:, :, :], mask=pre_covariate[:, :, :] == -9999.00000)

    input_images_pre0_ma = np.ma.array(pre_x[:, :, :, :, :],
                                         mask=pre_x[:, :, :, :, :] == -9999.0)
    # print(input_images_test0_ma.shape)
    input_covariate_pre0_ma = np.ma.array(pre_covariate[:, :, :],
                                         mask=pre_covariate[:, :, :] == -9999.0)
    # print(input_covariate_pre0_ma.shape)

    # print(input_covariate_test0_ma.shape)
    # 分情况进行标准化
    if is_single_norm == True:
        # 对所有天所有站点 求各个波段的均值和标准差
        # mean_pre = a.mean(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)  # （7,1,1）
        # std_pre = a.std(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        # Extract the values from the DataFrame into arrays and reshape
        mean_pre = mean_array
        std_pre = stdDev_array
        # max_pre = a.max(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        # min_pre = a.min(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        # print(mean_pre, std_pre,max_pre,min_pre)
        input_images_pre_norm0[:,:,:, :, :] = (input_images_pre0_ma[:,:,:, :, :] - mean_pre[:, :, :]) / std_pre[:,:,:]  # （540，34,8,1）
        # print(input_images_pre_norm0[2, 50, 3, :, :])
        # print(input_images_test_norm0)
        # 对其他协变量进行标准化
        mean_pre_cov = b.mean(axis=(0, 1))[1:15]
        std_pre_cov = b.std(axis=(0, 1))[1:15]
        max_pre_cov = b.max(axis=(0, 1))[1:15]
        min_pre_cov = b.min(axis=(0, 1))[1:15]
        input_covariate_pre_norm0[:, :, 1:15] = (input_covariate_pre0_ma[:, :, 1:15] - mean_pre_cov) / std_pre_cov
        # print(input_covariate_pre_norm0)
        # print(input_covariate_test_norm0)
    else:
        mean_pre = a.mean(axis=0)  # （7,50,50）
        # print(mean_pre)
        std_pre = a.std(axis=0)
        input_images_pre_norm0[:,:,:, :, :] = (input_images_pre0_ma[:,:,:, :, :] - mean_pre[:, :, :]) / std_pre[:,:,:]  # （540，34,8,1）
        # print(input_images_pre_norm0)

        # print(input_images_test_norm0)
        # 对其他协变量进行标准化
        mean_pre_cov = b.mean(axis=0)  # （7,1,1）
        std_pre_cov = b.std(axis=0)
        max_pre_cov = b.max(axis=0)
        min_pre_cov = b.min(axis=0)
        input_covariate_pre_norm0[:, :, :] = (input_covariate_pre0_ma[:, :, :] - mean_pre_cov[:, :,:]) / std_pre_cov[:, :,:]  # （540，34,8,1）
        # print(input_covariate_pre_norm0)
        # print(input_covariate_test_norm0)
    # 将影像数组展开为一维
    mean_pre = mean_pre.flatten()
    std_pre = std_pre.flatten()
    # max_pre = max_pre.flatten()
    # min_pre = min_pre.flatten()
    # 创建字典，键为列名，值为对应的数组
    image_eigenvalues = {
        'mean_pre': mean_pre,
        'std_pre': std_pre,
        # 'max_pre': max_pre,
        # 'min_pre': min_pre
    }
    # 创建 DataFrame 对象
    df = pd.DataFrame(image_eigenvalues)
    # 设置输出路径和文件名
    statistic_DIR = './region model/sites_statistic/'
    if not os.path.isdir(statistic_DIR):
        # 创建文件夹
        os.makedirs(statistic_DIR)
        print(f"已创建文件夹：{statistic_DIR}")
    else:
        print(f"文件夹已存在：{statistic_DIR}")
    # 将 DataFrame 为 CSV 文件
    df.to_csv(os.path.join(statistic_DIR, 'Image_mean_std_RLM.csv'), index=False)
    return input_images_pre_norm0,input_covariate_pre_norm0,mean_pre,std_pre,mean_pre_cov,std_pre_cov


# def pre_construct_composite_norm(pre_x,pre_covariate,is_single_norm=False):
#     # 复制数据
#     input_images_pre_norm0 = pre_x.copy()
#     input_covariate_pre_norm0 = pre_covariate.copy()
#     # ！！！注意对于-9999值的处理：值为-9999的不应该参与均值和标准差的计算
#     # 但由于前面已经将空值、为云 雪等的值设为了-9999，所以不用额外设置，后续处理需要尤其注意
#     # 引入屏蔽数组，屏蔽无效或者不完整的数据
#
#     # 将测试集与训练集合并到一起，对应的均值和标准差即为所有的数据求得的
#     a = np.ma.array(pre_x[:, :, :, :, :], mask=pre_x[:, :, :, :, :] == -9999.00000)
#     b = np.ma.array(pre_covariate[:, :, :], mask=pre_covariate[:, :, :] == -9999.00000)
#
#     input_images_pre0_ma = np.ma.array(pre_x[:, :, :, :, :],
#                                          mask=pre_x[:, :, :, :, :] == -9999.0)
#     # print(input_images_pre0_ma.shape)
#
#     # print(input_images_test0_ma.shape)
#     input_covariate_pre0_ma = np.ma.array(pre_covariate[:, :, :],
#                                          mask=pre_covariate[:, :, :] == -9999.0)
#     # print(input_covariate_pre0_ma.shape)
#
#     # print(input_covariate_test0_ma.shape)
#     # 分情况进行标准化
#     if is_single_norm == True:
#         # 对所有天所有站点 求各个波段的均值和标准差
#         mean_pre = a.mean(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)  # （7,1,1）
#         std_pre = a.std(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
#         # max_pre = a.max(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
#         # min_pre = a.min(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
#         # print(mean_pre, std_pre,max_pre,min_pre)
#         input_images_pre_norm0[:,:,:, :, :] = (input_images_pre0_ma[:,:,:, :, :] - mean_pre[:, :, :]) / std_pre[:,:,:]  # （540，34,8,1）
#         # print(input_images_pre_norm0[2, 50, 3, :, :])
#         # print(input_images_test_norm0)
#         # 对其他协变量进行标准化
#         mean_pre_cov = b.mean(axis=(0, 1))[1:15]
#         std_pre_cov = b.std(axis=(0, 1))[1:15]
#         max_pre_cov = b.max(axis=(0, 1))[1:15]
#         min_pre_cov = b.min(axis=(0, 1))[1:15]
#         input_covariate_pre_norm0[:, :, 1:15] = (input_covariate_pre0_ma[:, :, 1:15] - mean_pre_cov) / std_pre_cov
#         # print(input_covariate_pre_norm0)
#         # print(input_covariate_test_norm0)
#     else:
#         mean_pre = a.mean(axis=0)  # （7,50,50）
#         # print(mean_pre)
#         std_pre = a.std(axis=0)
#         input_images_pre_norm0[:,:,:, :, :] = (input_images_pre0_ma[:,:,:, :, :] - mean_pre[:, :, :]) / std_pre[:,:,:]  # （540，34,8,1）
#         # print(input_images_pre_norm0)
#
#         # print(input_images_test_norm0)
#         # 对其他协变量进行标准化
#         mean_pre_cov = b.mean(axis=0)  # （7,1,1）
#         std_pre_cov = b.std(axis=0)
#         max_pre_cov = b.max(axis=0)
#         min_pre_cov = b.min(axis=0)
#         input_covariate_pre_norm0[:, :, :] = (input_covariate_pre0_ma[:, :, :] - mean_pre_cov[:, :,:]) / std_pre_cov[:, :,:]  # （540，34,8,1）
#         # print(input_covariate_pre_norm0)
#         # print(input_covariate_test_norm0)
#     # 将影像数组展开为一维
#     mean_pre = mean_pre.flatten()
#     std_pre = std_pre.flatten()
#     # max_pre = max_pre.flatten()
#     # min_pre = min_pre.flatten()
#     # 创建字典，键为列名，值为对应的数组
#     image_eigenvalues = {
#         'mean_pre': mean_pre,
#         'std_pre': std_pre,
#         # 'max_pre': max_pre,
#         # 'min_pre': min_pre
#     }
#     # 创建 DataFrame 对象
#     df = pd.DataFrame(image_eigenvalues)
#     # 设置输出路径和文件名
#     statistic_DIR = './data/sites_statistic/'
#     if not os.path.isdir(statistic_DIR):
#         # 创建文件夹
#         os.makedirs(statistic_DIR)
#         print(f"已创建文件夹：{statistic_DIR}")
#     else:
#         print(f"文件夹已存在：{statistic_DIR}")
#     # 将 DataFrame 为 CSV 文件
#     df.to_csv(os.path.join(statistic_DIR, 'Image_eigenvalues.csv'), index=False)
#     return input_images_pre_norm0,input_covariate_pre_norm0,mean_pre,std_pre,mean_pre_cov,std_pre_cov



def mul_match_x_y_modis(x_image_array_all,ref=True,
                        lai=True,era5_meteorology=True,ref_quality=0, ref_filledvalue=0,
                        lai_filledvalue=0, lai_quality=0,era5_mete_filledvalue=0):
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

    # x_y_indices = x1_nonfilled + x1_filled + x1_all + x1_valid_quality_good + x1_valid_quality_bad + x1_quality_all +\
    #                         x2_nonfilled + x2_filled + x2_all + x2_valid_quality_good + x2_valid_quality_bad + x2_qaulity_all + \
    #                         x3_valid + x3_invalid + x3_all + y_valid + y_invalid + y_all
    # print(len(x_y_indices))
    all_lists = [x1_nonfilled, x1_filled, x1_all, x1_valid_quality_good, x1_valid_quality_bad, x1_quality_all,
                 x2_nonfilled, x2_filled, x2_all, x2_valid_quality_good, x2_valid_quality_bad, x2_qaulity_all,
                 x3_valid, x3_invalid, x3_all, x4_valid, x4_invalid, x4_all,y_valid, y_invalid, y_all]

    # 使用reduce函数找出同时出现在所有非空列表中的元组
    common_tuples = reduce(lambda x, y: set(x) & set(y), filter(None, all_lists))
    print(len(common_tuples))
    # x_image_array_all[963, 196, :7, :, :]
    sorted_x_all_y_valid_indices = sorted(common_tuples, key=lambda x: (x[0], x[1]))

    # 所有site year的数据
    x_image_array_list = []
    y_image_array_list = []
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
        for index in indices:
            x_image_array_selected = x_image_array_all[index]
            # y_image_array_selected = y_image_array_all[index]
            one_site_year_x_data.append(x_image_array_selected)
            # one_site_year_y_data.append(y_image_array_selected)
        x_image_array_list.append(np.array(one_site_year_x_data))
        y_image_array_list.append(np.array(one_site_year_y_data))
    new_x_array = x_image_array_list
    new_y_array = y_image_array_list

    # 根据new_array最大的time series 进行padding
    # 同理，对x_影像进行padding
    max_shape = max([arrx.shape for arrx in new_x_array])
    # 对每个数组进行填充
    padded_x_array = []
    for arrx in new_x_array:
        pad_width = [(0, max_shape[i] - arrx.shape[i]) for i in range(len(max_shape))]
        padded_arr = np.pad(arrx, pad_width, mode='constant', constant_values=-9999)
        padded_x_array.append(padded_arr)

    x_image_array = np.stack(padded_x_array)
    # 过滤掉所有doy的y值全为-9999的siteyear
    x_all_valid_indices11 = []
    x_all_invalid_indices11 = []
    for m in range(x_image_array.shape[0]):
        if np.any(x_image_array[m] != -9999):
            x_all_valid_indices11.append(m)
        else:
            x_all_invalid_indices11.append(m)
    x_image_array11 = x_image_array[x_all_valid_indices11]


    selected_bands = [0, 1, 2, 3, 4, 5, 6]  # Bands for reflectance
    if ref == True:
        if lai == True:
            # 若只使用fpar
            selected_bands.extend([14,15])   # Bands for reflectance and LAI
            # # 若使用fpar和lai
            # selected_bands.extend([14,15])  # Bands for reflectance and LAI
        if era5_meteorology == True:
             # Bands for reflectance and ERA5 meteorology
             selected_bands.extend([17, 18, 19, 20, 21, 22])
    x_image_array_44 = x_image_array11[:, :, selected_bands, :, :]

    # 删除23个特征里面的第2个特征（即 第2维度的第一个index,为了和之前的数组保持一致，且删掉的是经度，倒数第2个也是，所以不影响）
    # y_image_array_44 = np.delete(y_image_array33, 1, axis=2)
    return x_image_array_44