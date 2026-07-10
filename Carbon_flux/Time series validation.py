import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science', 'ieee'])
# plt.style.use(['science','bright'])

if __name__ == "__main__":
    inputdata_path = './data/sites_output/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RLM_predict_1.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RLM_true_1.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RLM_sites_1.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RLM_x_mask_1.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RLM_y_qa_1.npy', allow_pickle=True)
    DL_y_qa[:, :, 3]
    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RLM_NEE_predict_1.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RLM_NEE_true_1.npy', allow_pickle=True)
    RF_NEE_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_1.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RLM_GPP_predict_1.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RLM_GPP_true_1.npy', allow_pickle=True)
    RF_GPP_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_1.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RLM_RECO_predict_1.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RLM_RECO_true_1.npy', allow_pickle=True)
    RF_RECO_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_1.npy', allow_pickle=True)
    # XGB three
    XGB_NEE_predict = np.load(inputdata_path + 'XGB_RLM_NEE_predict_1.npy', allow_pickle=True)
    XGB_NEE_true = np.load(inputdata_path + 'XGB_RLM_NEE_true_1.npy', allow_pickle=True)
    XGB_NEE_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_1.npy', allow_pickle=True)

    XGB_GPP_predict = np.load(inputdata_path + 'XGB_RLM_GPP_predict_1.npy', allow_pickle=True)
    XGB_GPP_true = np.load(inputdata_path + 'XGB_RLM_GPP_true_1.npy', allow_pickle=True)
    XGB_GPP_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_1.npy', allow_pickle=True)

    XGB_RECO_predict = np.load(inputdata_path + 'XGB_RLM_RECO_predict_1.npy', allow_pickle=True)
    XGB_RECO_true = np.load(inputdata_path + 'XGB_RLM_RECO_true_1.npy', allow_pickle=True)
    XGB_RECO_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_1.npy', allow_pickle=True)

    # 获取唯一的站点ID
    unique_site_ids = np.unique(np.array(RF_RECO_sites[:, 0]))
    # 从这些站点中随机选择几个站点
    # 从unique_site_ids中随机选择三个站点
    # np.random.seed(0)  # 设置随机种子为0
    selected_sites = np.random.choice(unique_site_ids, 3, replace=False)
    # # ********************************************************************************
    # # 随机森林的结果
    # 找到selected_sites在站点名称中的索引
    RF_site_indices = np.where(np.isin(RF_RECO_sites[:, 0], selected_sites))[0]
    # 从RF_All_true_array中选出selected_sites对应的所有数据
    RF_selected_data = RF_RECO_sites[RF_site_indices]

    # 选择对应的年份
    sites_array = RF_selected_data
    # # 创建一个空字典来存储每个站点每年的数据量
    site_year_data = {}

    # 计算每个站点每年的数据量
    for entry in sites_array:
        site = entry[0]
        year = entry[3]
        if (site, year) in site_year_data:
            site_year_data[(site, year)] += 1
        else:
            site_year_data[(site, year)] = 1

    # # 找到每个站点数据量最多的年份
    max_data_years = {}
    for (site, year), count in site_year_data.items():
        if site in max_data_years:
            if count > max_data_years[site][1]:
                max_data_years[site] = (year, count)
        else:
            max_data_years[site] = (year, count)

    RF_selected_indices = [index for index, entry in enumerate(RF_RECO_sites) if
                           entry[0] in selected_sites and entry[3] == max_data_years[entry[0]][0]]

    RF_true = RF_RECO_sites[RF_selected_indices, 6:9]
    RF_pred = np.stack((RF_NEE_predict, RF_GPP_predict, RF_RECO_predict), axis=1)
    RF_pred = RF_pred[RF_selected_indices, :]
    RF_sites = RF_RECO_sites[RF_selected_indices][:, [0, 3, 4]]
    # 将三组RF 横向拼接成一共数组
    RF_merged_array = np.concatenate((RF_sites, RF_true, RF_pred), axis=1)
    # 将RF_merged_array转为df,列名依次为Site Year Doy RF_true_NEE RF_true_GPP RF_true_RECO RF_pred_NEE RF_pred_GPP RF_pred_RECO
    # 将RF_merged_array转换为DataFrame
    RF_df = pd.DataFrame(RF_merged_array,
                         columns=["Site", "Year", "Doy", "RF_true_NEE", "RF_true_GPP", "RF_true_RECO", "RF_pred_NEE",
                                  "RF_pred_GPP", "RF_pred_RECO"])
    # 根据Site和Year进行分组
    RF_grouped_data = RF_df.groupby(['Site', 'Year'], sort=False)

    XGB_true = XGB_RECO_sites[RF_selected_indices, 6:9]
    XGB_pred = np.stack((XGB_NEE_predict, XGB_GPP_predict, XGB_RECO_predict), axis=1)
    XGB_pred = XGB_pred[RF_selected_indices, :]
    XGB_sites = XGB_RECO_sites[RF_selected_indices][:, [0, 3, 4]]
    # 将三组RF 横向拼接成一共数组
    XGB_merged_array = np.concatenate((XGB_sites, XGB_true, XGB_pred), axis=1)
    # 将RF_merged_array转为df,列名依次为Site Year Doy RF_true_NEE RF_true_GPP RF_true_RECO RF_pred_NEE RF_pred_GPP RF_pred_RECO
    # 将RF_merged_array转换为DataFrame
    XGB_df = pd.DataFrame(XGB_merged_array,
                         columns=["Site", "Year", "Doy", "XGB_true_NEE", "XGB_true_GPP", "XGB_true_RECO", "XGB_pred_NEE",
                                  "XGB_pred_GPP", "XGB_pred_RECO"])
    # 根据Site和Year进行分组
    XGB_grouped_data = XGB_df.groupby(['Site', 'Year'], sort=False)

    # # 深度学习数据
    DL_selected_data = []
    DL_selected_indices = []
    for site, (year, _) in max_data_years.items():  # 获取每个站点数据量最多的年份
        for idx, entry in enumerate(DL_All_true):
            if entry[0, 0] == site and entry[0, 3] == year:  # 检查站点和年份是否匹配
                DL_selected_data.append(entry)
                DL_selected_indices.append(idx)

    DL_selected_data = np.array(DL_selected_data)
    # DL_selected_indices = np.array(DL_selected_indices)

    DL_true = DL_All_true[DL_selected_indices,:, 6:9]
    DL_pred = DL_Y_predict[DL_selected_indices, :,:]
    DL_sites = DL_All_true[DL_selected_indices][:,:,[0,3,4]]
    print('hhh')

    # ******************************************************
    # 这3个站点，每个站点选择一年可视化三个变量的分布情况
    # 对于每个选定的站点，选择数据量最多的年份对应的数据
    num_site_year = DL_true.shape[0]
    # # Create the subplots with the specified layout
    num_site_year = DL_true.shape[0]
    plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman', 'mathtext.fontset': 'custom',
                         'axes.unicode_minus': False})

    # Create the subplots with the specified layout
    fig, axes = plt.subplots(nrows=num_site_year, ncols=3, figsize=(12, 14), dpi=300)
    # plt.style.use(['science','ieee'])
    i = 0
    for RF_name, RF_group in RF_grouped_data:
        site_name = RF_name[0]  # 获取站点名称
        site_year = RF_name[1] # 获取站点年份
        print(RF_name)
        print(RF_group)
        for j in range(0,DL_true.shape[2]):
            axes[i, j].plot(RF_group['Doy'], RF_group.iloc[:, j + 3],label=RF_group.columns[j + 3].replace('RF_true_', 'Ground '))
            axes[i, j].plot(RF_group['Doy'], RF_group.iloc[:, j + 6],
                            label=RF_group.columns[j + 6].replace('RF_pred_', 'RF pred '))
            axes[i, j].plot(RF_group['Doy'], XGB_df[XGB_df["Site"] == site_name].iloc[:, j + 6],
                            label=RF_group.columns[j + 6].replace('RF_pred_', 'XGB pred '))
            DL_mask = DL_true[i, :, j] == -9999
            DL_predij = DL_pred[i, :, j][~DL_mask]
            axes[i, j].plot(RF_group['Doy'], DL_predij, label=RF_group.columns[j + 6].replace('RF_pred_', 'DL pred '))
            axes[i, j].legend(loc='upper right')  # 在每个子图中添加图例
            if i == 2:
                axes[2, j].set_xlabel('Doy', fontsize=12)  # 在每列的底部行添加横轴标签
            else:
                continue
        # 每个站点遍历完之后设置三个站点子图的标题为站点名称+年份，居中
        axes[i, 0].set_ylabel('Unit (g C m$^{-2}$ d$^{-1}$)', fontsize=12)  # 在每行的最左侧列添加纵轴标签
        axes[i, 1].set_title(site_name + ' ' + str(site_year), loc='center', pad=16, fontdict={'fontsize': 14})# 将年份转为字符串并与站点名组合
        i += 1
    plot_title = 'Comparison of time series of DL and RF model predictions with ground-based observations'
    plt.suptitle(plot_title, y=0.99,fontsize=15)
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    resluts_path = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    plt.savefig(resluts_path + plot_title +'3'+ '.png', dpi=600)
    plt.show()