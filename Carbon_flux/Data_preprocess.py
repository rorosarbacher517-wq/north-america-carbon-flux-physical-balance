import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# 定义一个data load函数 加载x1 变量数据
rootpath = os.getcwd()
print(rootpath)

def random_split(total_n, split_n, y_array):
    """
    create split index by splitting total_n into train and test based on split_n
    test: orders == 0
    train: order != 0
    """
    sample_n = np.ceil(total_n / split_n).astype(int)
    SPLIT_DIR = "./region model/split/"
    if not os.path.isdir(SPLIT_DIR):
        # create directory
        os.makedirs(SPLIT_DIR)
        print(f"Created directory: {SPLIT_DIR}")
    else:
        print(f"Directory already exists: {SPLIT_DIR}")
    file_index = SPLIT_DIR + "index.total_n" + str(total_n) + ".for.random.txt"
    if os.path.isfile(file_index):
        print('File already exists! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i == 0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        orders = orders[range(total_n)]
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    # Create a dictionary to map site to order
    site_order_dict = {}
    for i, order in enumerate(orders):
        site = y_array[i, 0, 0]  # Assuming the site information is in y_array[:, 0, 0]
        site_order_dict[site] = order
    # Assign order to each site-year pair in y_array
    for i in range(y_array.shape[0]):
        site = y_array[i, 0, 0]
        y_array[i, :, 0] = site_order_dict[site]
    # print(site_order_dict)
    return y_array

############################ 5折交叉验证划分
# 将字符类型转换为char_to_number字典中的数字
def convert_to_numerical(array, char_to_number):
    numerical_array = np.where(array == -9999, -9999, np.vectorize(char_to_number.get)(array)) # Using vectorize for element-wise mapping
    return numerical_array

# Create a custom LabelEncoder class to handle -9999 values
class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(y[y != '-9999'])  # Only fit with non -9999 values
        return self

    def transform(self, y):
        y_transformed = y.copy()
        y_transformed[y != '-9999'] = super().transform(y[y != '-9999'])  # Transform non -9999 values
        return y_transformed


# 多倍交叉验证划分训练集和测试集，按植被类型输出
def get_cross_validation_training_test_vegetation(x_array,y_array,covariate_array):
    """used construct_composite_train_test to get train and test"""
    MODEL_DIR = './data/sites_split/'
    if not os.path.isdir(MODEL_DIR):
        # 创建文件夹
        os.makedirs(MODEL_DIR)
        print(f"已创建文件夹：{MODEL_DIR}")
    else:
        print(f"文件夹已存在：{MODEL_DIR}")
    # print(x_array.shape)
    # 获取唯一的站点ID
    unique_site_ids = np.unique(np.array(y_array[:, 0, 0]))
    # 将所有站点‘str’用数字来代替，创建一个字典，字典为站点名和数字相匹配，对站点名进行编码
    vegetation_array = np.array(y_array[:, :, 15])
    # Reshape the arrays for encoding
    vegetation_array_reshaped = vegetation_array.reshape(-1, 1)
    # Convert the sites_array_reshaped to strings
    vegetation_array_str = vegetation_array_reshaped.astype(str)
    # Initialize the custom LabelEncoder
    label_encoder = CustomLabelEncoder()
    # Fit and transform the training data
    vegetation_array_with_numerical_labels = label_encoder.fit_transform(vegetation_array_str)
    vegetation_array_numerical_array = vegetation_array_with_numerical_labels.reshape(vegetation_array.shape)
    # Replace the values equal to 0 with -9999
    vegetation_array_numerical_array = np.where(vegetation_array_numerical_array == 0, -9999, vegetation_array_numerical_array)
    # # Verify the shape of the updated array
    # print("Shape of sites_array_numerical_array:", sites_array_numerical_array.shape)
    # Merge sites_array and sites_array_numerical_array into a single array
    vegetation_id_array = np.stack((vegetation_array, vegetation_array_numerical_array), axis=-1)

    all_indices = np.arange(y_array.shape[0])
    # 使用K折交叉验证来划分站点ID，把所有站点划分为k份
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建5折交叉验证对象
    # 通过交叉验证来索引站点ID
    cross_validation_results = []
    k = 0
    for train_index, test_index in kfold.split(unique_site_ids):
        # 第k折的测试集站点
        k_test_site_name = unique_site_ids[test_index]
        # 将k_test_site_name转换成NumPy数组
        test_site_ids_array = np.array(k_test_site_name)
        # 判断y_array[:, :, 0]中的元素是否在train_site_ids_array中,在此之前，x与y是一一对应的
        test_mask = np.isin(y_array[:, 0, 0], test_site_ids_array)
        # 找到满足训练集条件的元素的索引
        index_test = np.where(test_mask)[0]
        # print(index_test)
        # 找到在 all_indices 中不在 train_indices 中的索引
        index_train = np.setdiff1d(all_indices, index_test)
        # print(index_train)
        # 根据index划分训练集和测试集,后面补上经纬度、气象等辅助数据
        trainx = np.array(x_array[index_train]).astype(np.float32)
        # print(trainx.shape)
        # trainy = np.array(y_array[index_train][:,:,4])
        trainy = np.array(y_array[index_train][:, :, 5:8])
        RF_trainy = np.array(y_array[index_train][:, :, 5:8])
        # trainy_reshaped = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 3))
        # print(trainy.shape)
        train_covariate = np.array(covariate_array[index_train]).astype(np.float32)
        # print(train_covariate.shape)
        testx = np.array(x_array[index_test]).astype(np.float32)
        # print(testx.shape)
        # testy = np.array(y_array[index_test][:,:,4])
        testy = np.array(y_array[index_test][:, :, 5:8])
        RF_testy = np.array(y_array[index_test][:, :, 5:8])
        # testy_reshaped = np.reshape(testy, (testy.shape[0], testy.shape[1], 3))
        # print(testy.shape)
        test_covariate = np.array(covariate_array[index_test]).astype(np.float32)
        # print(test_covariate.shape)
        # 根据训练集和测试集划分结果，将站点和站点encode分别划分开来
        train_vegetation_id_array = np.array(vegetation_id_array[index_train])
        test_vegetation_id_array = np.array(vegetation_id_array[index_test])

        # # 根据索引将训练集和测试集的站点输出保存为txt,一共三列 一列是站点名，两列是经度和纬度
        # # 训练集索引和站点集
        train_df = y_array[:, 0, [0, 19, 20]][index_train]
        test_df = y_array[:, 0, [0, 19, 20]][index_test]
        # # 创建要保存的数据数组
        train_data = np.column_stack((index_train, train_df))
        test_data = np.column_stack((index_test, test_df))
        data_columns = ['Index', 'Site_Id', 'Latitude(degrees)', 'Longitude(degrees)']
        # 根据索引获取训练集数据
        train_sites = pd.DataFrame(data=train_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        train_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        train_sites.reset_index(drop=True, inplace=True)
        test_sites = pd.DataFrame(data=test_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        test_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        test_sites.reset_index(drop=True, inplace=True)
        # 计算训练集和测试集的站点数
        train_sites_counts = train_sites['Site_Id'].value_counts()
        test_sites_counts = test_sites['Site_Id'].value_counts()
        # 将数据保存到文件
        # 创建站点文件夹
        # 将 DataFrame 保存为 CSV 文件
        train_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_train_'+ str(k)+'.csv'), index=False)
        test_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_test_'+ str(k)+ '.csv'), index=False)
        # 将每次交叉验证的结果保存到cross_validation_results列表中
        cross_validation_results.append({
            'index_train':index_train,
            'index_test':index_test,
            'trainx': trainx,
            'trainy': trainy,
            'testx': testx,
            'testy': testy,
            'train_covariate': train_covariate,
            'test_covariate': test_covariate,
            'train_sites_counts':train_sites_counts,
            'test_sites_counts':test_sites_counts,
            'RF_trainy':RF_trainy,
            'RF_testy':RF_testy,
            'train_vegetation_id_array':train_vegetation_id_array,
            'test_vegetation_id_array':test_vegetation_id_array
        })
        k += 1
    return cross_validation_results

# 多倍交叉验证划分训练集和测试集，按站点输出
def get_cross_validation_training_test_sites(x_array,y_array,covariate_array):
    """used construct_composite_train_test to get train and test"""
    MODEL_DIR = './data/sites_split/'
    if not os.path.isdir(MODEL_DIR):
        # 创建文件夹
        os.makedirs(MODEL_DIR)
        print(f"已创建文件夹：{MODEL_DIR}")
    else:
        print(f"文件夹已存在：{MODEL_DIR}")
    # print(x_array.shape)
    # 获取唯一的站点ID
    unique_site_ids = np.unique(np.array(y_array[:, 0, 0]))
    # 将所有站点‘str’用数字来代替，创建一个字典，字典为站点名和数字相匹配，对站点名进行编码
    sites_array = np.array(y_array[:, :, 0])
    # Reshape the arrays for encoding
    sites_array_reshaped = sites_array.reshape(-1, 1)
    # Convert the sites_array_reshaped to strings
    sites_array_str = sites_array_reshaped.astype(str)
    # Initialize the custom LabelEncoder
    label_encoder = CustomLabelEncoder()
    # Fit and transform the training data
    sites_array_with_numerical_labels = label_encoder.fit_transform(sites_array_str)
    sites_array_numerical_array = sites_array_with_numerical_labels.reshape(sites_array.shape)
    # Replace the values equal to 0 with -9999
    sites_array_numerical_array = np.where(sites_array_numerical_array == 0, -9999, sites_array_numerical_array)
    # # Verify the shape of the updated array
    # print("Shape of sites_array_numerical_array:", sites_array_numerical_array.shape)
    # Merge sites_array and sites_array_numerical_array into a single array
    site_id_array = np.stack((sites_array, sites_array_numerical_array), axis=-1)

    all_indices = np.arange(y_array.shape[0])
    # 使用K折交叉验证来划分站点ID，把所有站点划分为k份
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建5折交叉验证对象
    # 通过交叉验证来索引站点ID
    cross_validation_results = []
    k = 0
    for train_index, test_index in kfold.split(unique_site_ids):
        # 第k折的测试集站点
        k_test_site_name = unique_site_ids[test_index]
        # 将k_test_site_name转换成NumPy数组
        test_site_ids_array = np.array(k_test_site_name)
        # 判断y_array[:, :, 0]中的元素是否在train_site_ids_array中,在此之前，x与y是一一对应的
        test_mask = np.isin(y_array[:, 0, 0], test_site_ids_array)
        # 找到满足训练集条件的元素的索引
        index_test = np.where(test_mask)[0]
        # print(index_test)
        # 找到在 all_indices 中不在 train_indices 中的索引
        index_train = np.setdiff1d(all_indices, index_test)
        # print(index_train)
        # 根据index划分训练集和测试集,后面补上经纬度、气象等辅助数据
        trainx = np.array(x_array[index_train]).astype(np.float32)
        # print(trainx.shape)
        # trainy = np.array(y_array[index_train][:,:,4])
        trainy = np.array(y_array[index_train][:, :, 5:8])
        RF_trainy = np.array(y_array[index_train][:, :, 5:8])
        # trainy_reshaped = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 3))
        # print(trainy.shape)
        train_covariate = np.array(covariate_array[index_train]).astype(np.float32)
        # print(train_covariate.shape)
        testx = np.array(x_array[index_test]).astype(np.float32)
        # print(testx.shape)
        # testy = np.array(y_array[index_test][:,:,4])
        testy = np.array(y_array[index_test][:, :, 5:8])
        RF_testy = np.array(y_array[index_test][:, :, 5:8])
        # testy_reshaped = np.reshape(testy, (testy.shape[0], testy.shape[1], 3))
        # print(testy.shape)
        test_covariate = np.array(covariate_array[index_test]).astype(np.float32)
        # print(test_covariate.shape)
        # 根据训练集和测试集划分结果，将站点和站点encode分别划分开来
        train_site_id_array = np.array(site_id_array[index_train])
        test_site_id_array = np.array(site_id_array[index_test])

        # # 根据索引将训练集和测试集的站点输出保存为txt,一共三列 一列是站点名，两列是经度和纬度
        # # 训练集索引和站点集
        train_df = y_array[:, 0, [0, 19, 20]][index_train]
        test_df = y_array[:, 0, [0, 19, 20]][index_test]
        # # 创建要保存的数据数组
        train_data = np.column_stack((index_train, train_df))
        test_data = np.column_stack((index_test, test_df))
        data_columns = ['Index', 'Site_Id', 'Latitude(degrees)', 'Longitude(degrees)']
        # 根据索引获取训练集数据
        train_sites = pd.DataFrame(data=train_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        train_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        train_sites.reset_index(drop=True, inplace=True)
        test_sites = pd.DataFrame(data=test_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        test_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        test_sites.reset_index(drop=True, inplace=True)
        # 计算训练集和测试集的站点数
        train_sites_counts = train_sites['Site_Id'].value_counts()
        test_sites_counts = test_sites['Site_Id'].value_counts()
        # 将数据保存到文件
        # 创建站点文件夹
        # 将 DataFrame 保存为 CSV 文件
        train_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_train_'+ str(k)+'_v1.csv'), index=False)
        test_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_test_'+ str(k)+ '_v1.csv'), index=False)
        # 将每次交叉验证的结果保存到cross_validation_results列表中
        cross_validation_results.append({
            'index_train': index_train,
            'index_test': index_test,
            'trainx': trainx,
            'trainy': trainy,
            'testx': testx,
            'testy': testy,
            'train_covariate': train_covariate,
            'test_covariate': test_covariate,
            'train_sites_counts': train_sites_counts,
            'test_sites_counts': test_sites_counts,
            'RF_trainy': RF_trainy,
            'RF_testy': RF_testy,
            'train_site_id_array': train_site_id_array,
            'test_site_id_array': test_site_id_array
        })
        k += 1
    return cross_validation_results

# 简单随机划分 训练集和测试集
def get_cross_validation_training_test(x_array,y_array,covariate_array):
    """used construct_composite_train_test to get train and test"""
    MODEL_DIR = './data/sites_split/'
    if not os.path.isdir(MODEL_DIR):
        # 创建文件夹
        os.makedirs(MODEL_DIR)
        print(f"已创建文件夹：{MODEL_DIR}")
    else:
        print(f"文件夹已存在：{MODEL_DIR}")
    # print(x_array.shape)
    # 获取唯一的站点ID
    unique_site_ids = np.unique(np.array(y_array[:, 0, 0]))
    # 创建包含所有索引的数组
    all_indices = np.arange(y_array.shape[0])
    # 使用K折交叉验证来划分站点ID，把所有站点划分为k份
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建5折交叉验证对象
    # 通过交叉验证来索引站点ID
    cross_validation_results = []
    k = 0
    for train_index, test_index in kfold.split(unique_site_ids):
        # 第k折的测试集站点
        k_test_site_name = unique_site_ids[test_index]
        # 将k_test_site_name转换成NumPy数组
        test_site_ids_array = np.array(k_test_site_name)
        # 判断y_array[:, :, 0]中的元素是否在train_site_ids_array中,在此之前，x与y是一一对应的
        test_mask = np.isin(y_array[:, 0, 0], test_site_ids_array)
        # 找到满足训练集条件的元素的索引
        index_test = np.where(test_mask)[0]
        # print(index_test)
        # 找到在 all_indices 中不在 train_indices 中的索引
        index_train = np.setdiff1d(all_indices, index_test)
        # print(index_train)
        # 根据index划分训练集和测试集,后面补上经纬度、气象等辅助数据
        trainx = np.array(x_array[index_train]).astype(np.float32)
        # print(trainx.shape)
        # trainy = np.array(y_array[index_train][:,:,4])
        trainy = np.array(y_array[index_train][:, :, 6:9])
        RF_trainy = np.array(y_array[index_train][:, :, 6:9])
        # trainy_reshaped = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 3))
        # print(trainy.shape)
        train_covariate = np.array(covariate_array[index_train]).astype(np.float32)
        # print(train_covariate.shape)
        testx = np.array(x_array[index_test]).astype(np.float32)
        # print(testx.shape)
        # testy = np.array(y_array[index_test][:,:,4])
        testy = np.array(y_array[index_test][:, :, 6:9])
        RF_testy = np.array(y_array[index_test][:, :, 6:9])
        # testy_reshaped = np.reshape(testy, (testy.shape[0], testy.shape[1], 3))
        # print(testy.shape)
        test_covariate = np.array(covariate_array[index_test]).astype(np.float32)
        # print(test_covariate.shape)
        test_array = np.array(y_array[index_test])
        # # 根据索引将训练集和测试集的站点输出保存为txt,一共三列 一列是站点名，两列是经度和纬度
        # # 训练集索引和站点集
        train_df = y_array[:, 0, [0, 1, 2]][index_train]
        test_df = y_array[:, 0, [0, 1, 2]][index_test]
        # # 创建要保存的数据数组
        train_data = np.column_stack((index_train, train_df))
        test_data = np.column_stack((index_test, test_df))
        data_columns = ['Index', 'Site_Id', 'Latitude(degrees)', 'Longitude(degrees)']
        # 根据索引获取训练集数据
        train_sites = pd.DataFrame(data=train_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        train_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        train_sites.reset_index(drop=True, inplace=True)
        test_sites = pd.DataFrame(data=test_data, columns=data_columns)
        # 去除重复值
        # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
        test_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
        # Reset Index column if required
        test_sites.reset_index(drop=True, inplace=True)
        # 计算训练集和测试集的站点数
        train_sites_counts = train_sites['Site_Id'].value_counts()
        test_sites_counts = test_sites['Site_Id'].value_counts()
        # 将数据保存到文件
        # 创建站点文件夹
        # 将 DataFrame 保存为 CSV 文件
        train_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_train_'+ str(k)+'.csv'), index=False)
        test_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_test_'+ str(k)+ '.csv'), index=False)
        # 将每次交叉验证的结果保存到cross_validation_results列表中
        cross_validation_results.append({
            'index_train':index_train,
            'index_test':index_test,
            'trainx': trainx,
            'trainy': trainy,
            'testx': testx,
            'testy': testy,
            'train_covariate': train_covariate,
            'test_covariate': test_covariate,
            'train_sites_counts':train_sites_counts,
            'test_sites_counts':test_sites_counts,
            'RF_trainy':RF_trainy,
            'RF_testy':RF_testy,
            'test_array':test_array
        })
        k += 1
    return cross_validation_results

# 简单按比例随机划分 训练集和测试集
def get_training_test_com2(x_array,y_array,covariate_array,proportion= 0.9,split_by_site = True):
    """used construct_composite_train_test to get train and test"""
    ## 从x_array中划分得到order和valid index
    print(x_array.shape)
    # ***************************************************************
    # 根据站点随机划分训练集和测试集
    if split_by_site == True:
        # Obtain unique site_ids
        unique_site_ids = np.unique(np.array(y_array[:, 0, 0]))
        unique_site_ids2 = set(y_array[:, 0, 0])
        total_n = unique_site_ids.shape[0]
        # Set the random seed for consistent results
        # 每次生成的随机数相同
        np.random.seed(0)
        # Shuffle the unique site IDs
        np.random.shuffle(unique_site_ids)
        # Define the fraction of samples for training set
        train_fraction = proportion
        # Calculate the number of samples for training set
        train_n = int(total_n * train_fraction)
        # Obtain the site_ids for training and test sets
        train_site_ids = np.random.choice(unique_site_ids, train_n, replace=False)
        # 将train_site_ids转换成NumPy数组
        train_site_ids_array = np.array(train_site_ids)
        # 判断y_array[:, :, 0]中的元素是否在train_site_ids_array中,在此之前，x与y是一一对应的
        train_mask = np.isin(y_array[:, 0, 0], train_site_ids_array)
        # 找到满足训练集条件的元素的索引
        index_train = np.where(train_mask)[0]
        # 创建包含所有索引的数组
        all_indices = np.arange(y_array.shape[0])
        # 找到在 all_indices 中不在 train_indices 中的索引
        index_test = np.setdiff1d(all_indices, index_train)
    # # 以下这种方式可以先过滤掉全为-9999的无效数据，保留有效index
    # valid_index = np.argwhere(x_array > 0)
    else:
        valid_index = np.arange(x_array.shape[0])
        orders = random_split(x_array.shape[0], split_n=10)
        # 根据站点—year进行划分
        if proportion == 0.9:
            index_train = np.logical_and(orders < 9, valid_index)
            index_test = np.logical_and(orders >= 9, valid_index)
        elif proportion == 0.8:
            index_train = np.logical_and(orders < 8, valid_index)
            index_test = np.logical_and(orders >= 8, valid_index)
        elif proportion == 0.7:
            index_train = np.logical_and(orders < 7, valid_index)
            index_test = np.logical_and(orders >= 7, valid_index)
        elif proportion == 0.6:
            index_train = np.logical_and(orders < 6, valid_index)
            index_test = np.logical_and(orders >= 6, valid_index)
        else:
            index_train = np.logical_and(orders == 0, valid_index)
            index_test = np.logical_and(orders != 0, valid_index)
    # 根据index划分训练集和测试集,后面补上经纬度、气象等辅助数据
    trainx = np.array(x_array[index_train]).astype(np.float32)
    # print(trainx.shape)
    # trainy = np.array(y_array[index_train][:,:,4])
    trainy = np.array(y_array[index_train][:, :, 6:9])
    RF_trainy = np.array(y_array[index_train][:, :, 6:9])
    # trainy_reshaped = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 3))
    # print(trainy.shape)
    train_covariate = np.array(covariate_array[index_train]).astype(np.float32)
    # print(train_covariate.shape)
    testx = np.array(x_array[index_test]).astype(np.float32)
    # print(testx.shape)
    # testy = np.array(y_array[index_test][:,:,4])
    testy = np.array(y_array[index_test][:, :, 6:9])
    RF_testy = np.array(y_array[index_test][:, :, 6:9])
    # testy_reshaped = np.reshape(testy, (testy.shape[0], testy.shape[1], 3))
    # print(testy.shape)
    test_covariate = np.array(covariate_array[index_test]).astype(np.float32)
    # print(test_covariate.shape)
    # # 根据索引将训练集和测试集的站点输出保存为txt,一共三列 一列是站点名，两列是经度和纬度
    # # 训练集索引和站点集
    train_df = y_array[:, 0, [0,1,2]][index_train]
    test_df = y_array[:, 0, [0,1,2]][index_test]
    # # 创建要保存的数据数组
    train_data = np.column_stack((index_train, train_df))
    test_data = np.column_stack((index_test, test_df))
    # data_columns = ['Site_Id','Sensor_id','Year','DoY','Month','NEE_uStar_f_daily','GPP_uStar_f_daily','Reco_uStar_daily','Rg_daily','PotRad_uStar_daily','Tair_f_daily','VPD_f_daily',
    #            'NEE_uStar_f_monthly','GPP_uStar_f_monthly','Reco_uStar_monthly','Vegetation_Abbreviation(IGBP)','Climate_Class_Abbreviation(Koeppen)','Mean_Average_Precipitation(mm)',
    #            'Mean_Average_Temperature(degreesC)','Latitude(degrees)','Longitude(degrees)','Elevation(m)']
    data_columns = ['Index','Site_Id','Latitude(degrees)','Longitude(degrees)']
    # 根据索引获取训练集数据
    train_sites = pd.DataFrame(data=train_data, columns=data_columns)
    # 去除重复值
    # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
    train_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
    # Reset Index column if required
    train_sites.reset_index(drop=True, inplace=True)
    test_sites = pd.DataFrame(data=test_data, columns=data_columns)
    # 去除重复值
    # Drop duplicates based on 'Site_Id', 'Latitude(degrees)', and 'Longitude(degrees)' columns
    test_sites.drop_duplicates(subset=['Site_Id'], inplace=True)
    # Reset Index column if required
    test_sites.reset_index(drop=True, inplace=True)
    # 计算训练集和测试集的站点数
    train_sites_counts = train_sites['Site_Id'].value_counts()
    test_sites_counts = test_sites['Site_Id'].value_counts()
    # 将数据保存到文件
    # 创建站点文件夹
    MODEL_DIR = './data/sites_split/'
    if not os.path.isdir(MODEL_DIR):
        # 创建文件夹
        os.makedirs(MODEL_DIR)
        print(f"已创建文件夹：{MODEL_DIR}")
    else:
        print(f"文件夹已存在：{MODEL_DIR}")
    # 将 DataFrame 保存为 CSV 文件
    train_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_train.csv'), index=False)
    test_sites.to_csv(os.path.join(MODEL_DIR, 'input_site_test.csv'), index=False)
    return index_train,index_test,trainx,trainy,testx,testy,train_covariate,test_covariate,train_sites_counts,test_sites_counts,RF_trainy,RF_testy

def construct_composite_train_test(trainx,testx,train_covariate,test_covariate,is_single_norm=False, is_train_test_com=True):
    # 复制数据
    input_images_train_norm0 = trainx.copy()
    input_images_test_norm0 = testx.copy()
    input_covariate_train_norm0 = train_covariate.copy()
    input_covariate_test_norm0 = test_covariate.copy()
    # ！！！注意对于-9999值的处理：值为-9999的不应该参与均值和标准差的计算
    # 但由于前面已经将空值、为云 雪等的值设为了-9999，所以不用额外设置，后续处理需要尤其注意
    # 引入屏蔽数组，屏蔽无效或者不完整的数据
    if is_train_test_com == True:
        # 将测试集与训练集合并到一起，对应的均值和标准差即为所有的数据求得的
        a = np.ma.array(np.concatenate((trainx, testx)),
                        mask=np.concatenate((trainx[:,:,:, :, :] == -9999.00000,
                                             testx[:,:,:, :, :] == -9999.00000)))
        b = np.ma.array(np.concatenate((train_covariate, test_covariate)),
                        mask=np.concatenate((train_covariate[:, :, :] == -9999.00000,
                                             test_covariate[:, :, :] == -9999.00000)))
    else:
        a = np.ma.array(np.concatenate(trainx),
                        mask=np.concatenate(trainx[:,:,:, :, :] == -9999.0))
        b = np.ma.array(np.concatenate(train_covariate),
                        mask=np.concatenate(train_covariate[:, :, :] == -9999.0))
    input_images_train0_ma = np.ma.array(trainx[:,:,:, :, :],
                                         mask=trainx[:,:,:, :, :] == -9999.0)
    # print(input_images_train0_ma.shape)
    input_images_test0_ma = np.ma.array(testx[:,:,:, :, :],
                                        mask=testx[:,:,:, :, :] == -9999.0)
    # print(input_images_test0_ma.shape)
    input_covariate_train0_ma = np.ma.array(train_covariate,
                                         mask=train_covariate[:, :, :] == -9999.0)
    # print(input_covariate_train0_ma.shape)
    input_covariate_test0_ma = np.ma.array(test_covariate[:, :, :],
                                        mask=test_covariate[:, :, :] == -9999.0)
    # print(input_covariate_test0_ma.shape)
    # 分情况进行标准化
    if is_single_norm == True:
        # 对所有天所有站点 求各个波段的均值和标准差
        mean_train = a.mean(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)  # （7,1,1）
        std_train = a.std(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        max_train = a.max(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        min_train = a.min(axis=(0,1,3,4)).reshape(a.shape[2], 1, 1)
        # print(mean_train, std_train,max_train,min_train)
        input_images_train_norm0[:,:,:, :, :] = (input_images_train0_ma[:,:,:, :, :] - mean_train[:, :, :]) / std_train[:,:,:]  # （540，34,8,1）
        # print(input_images_train_norm0[2, 50, 3, :, :])
        input_images_test_norm0[:,:,:, :, :] = (input_images_test0_ma[:,:,:, :, :] - mean_train[:, :, :]) / std_train[:,:, :]
        # print(input_images_test_norm0)
        # 对其他协变量进行标准化
        mean_train_cov = b.mean(axis=(0, 1))[1:15]
        std_train_cov = b.std(axis=(0, 1))[1:15]
        max_train_cov = b.max(axis=(0, 1))[1:15]
        min_train_cov = b.min(axis=(0, 1))[1:15]
        input_covariate_train_norm0[:, :, 1:15] = (input_covariate_train0_ma[:, :, 1:15] - mean_train_cov) / std_train_cov
        # print(input_covariate_train_norm0)
        input_covariate_test_norm0[:, :, 1:15] = (input_covariate_test0_ma[:, :, 1:15] - mean_train_cov) / std_train_cov
        # print(input_covariate_test_norm0)
    else:
        mean_train = a.mean(axis=0)  # （7,50,50）
        # print(mean_train)
        std_train = a.std(axis=0)
        input_images_train_norm0[:,:,:, :, :] = (input_images_train0_ma[:,:,:, :, :] - mean_train[:, :, :]) / std_train[:,:,:]  # （540，34,8,1）
        # print(input_images_train_norm0)
        input_images_test_norm0[:,:,:, :, :] = (input_images_test0_ma[:,:,:, :, :] - mean_train[:, :, :]) / std_train[:,:, :]
        # print(input_images_test_norm0)
        # 对其他协变量进行标准化
        mean_train_cov = b.mean(axis=0)  # （7,1,1）
        std_train_cov = b.std(axis=0)
        max_train_cov = b.max(axis=0)
        min_train_cov = b.min(axis=0)
        input_covariate_train_norm0[:, :, :] = (input_covariate_train0_ma[:, :, :] - mean_train_cov[:, :,:]) / std_train_cov[:, :,:]  # （540，34,8,1）
        # print(input_covariate_train_norm0)
        input_covariate_test_norm0[:, :, :] = (input_covariate_test0_ma[:, :, :] - mean_train_cov[:, :,:]) / std_train_cov[:, :, :]
        # print(input_covariate_test_norm0)
    # 将影像数组展开为一维
    mean_train = mean_train.flatten()
    std_train = std_train.flatten()
    max_train = max_train.flatten()
    min_train = min_train.flatten()
    # 创建字典，键为列名，值为对应的数组
    image_eigenvalues = {
        'mean_train': mean_train,
        'std_train': std_train,
        'max_train': max_train,
        'min_train': min_train
    }
    # 创建 DataFrame 对象
    df = pd.DataFrame(image_eigenvalues)
    # 设置输出路径和文件名
    statistic_DIR = './data/sites_statistic/'
    if not os.path.isdir(statistic_DIR):
        # 创建文件夹
        os.makedirs(statistic_DIR)
        print(f"已创建文件夹：{statistic_DIR}")
    else:
        print(f"文件夹已存在：{statistic_DIR}")
    # 将 DataFrame 为 CSV 文件
    df.to_csv(os.path.join(statistic_DIR, 'Image_eigenvalues.csv'), index=False)
    return input_images_train_norm0,input_images_test_norm0,input_covariate_train_norm0,input_covariate_test_norm0,mean_train,std_train,mean_train_cov,std_train_cov













