import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use(['science', 'ieee'])
# plt.style.use(['science','bright'])
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

if __name__ == "__main__":
    inputdata_path = './data/sites_results/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)
    DL_y_qa[:, :, 3]
    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RLM_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RLM_NEE_true_3.npy', allow_pickle=True)
    RF_NEE_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_3.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RLM_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RLM_GPP_true_3.npy', allow_pickle=True)
    RF_GPP_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_3.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RLM_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RLM_RECO_true_3.npy', allow_pickle=True)
    RF_RECO_sites = np.load(inputdata_path + 'RF_RLM_RECO_site_3.npy', allow_pickle=True)
    # XGB three
    XGB_NEE_predict = np.load(inputdata_path + 'XGB_RLM_NEE_predict_3.npy', allow_pickle=True)
    XGB_NEE_true = np.load(inputdata_path + 'XGB_RLM_NEE_true_3.npy', allow_pickle=True)
    XGB_NEE_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_3.npy', allow_pickle=True)

    XGB_GPP_predict = np.load(inputdata_path + 'XGB_RLM_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true = np.load(inputdata_path + 'XGB_RLM_GPP_true_3.npy', allow_pickle=True)
    XGB_GPP_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_3.npy', allow_pickle=True)

    XGB_RECO_predict = np.load(inputdata_path + 'XGB_RLM_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true = np.load(inputdata_path + 'XGB_RLM_RECO_true_3.npy', allow_pickle=True)
    XGB_RECO_sites = np.load(inputdata_path + 'XGB_RLM_RECO_site_3.npy', allow_pickle=True)

    # 获取唯一的站点ID
    # unique_site_ids = np.unique(np.array(RF_RECO_sites[:, 0]))
    # # 从这些站点中随机选择几个站点
    # # 从unique_site_ids中随机选择三个站点
    # np.random.seed(42)  # 设置随机种子为0
    # # selected_sites = np.random.choice(unique_site_ids, 10, replace=False)
    # # 以下站点是seed等于42时候的
    # # selected_sites = ['US-Rwe','US-Tw1','US-Ha2','US-A74','CA-ER1','CA-Cha','US-WCr','CA-TP4','US-Me6','US-KM3']
    # selected_sites = ['US-Ha2', 'US-Rwe', 'US-Me6']
    # # 获得这些站点的DL XGBoost RF 模型的数据
    # RF_selected_indices = [index for index, entry in enumerate(RF_RECO_sites) if
    #                        entry[0] in selected_sites]
    # RF_true = RF_RECO_sites[RF_selected_indices, 6:9]
    # RF_pred = np.stack((RF_NEE_predict, RF_GPP_predict, RF_RECO_predict), axis=1)
    # RF_pred = RF_pred[RF_selected_indices, :]
    # RF_sites = RF_RECO_sites[RF_selected_indices][:, [0, 3, 4]]
    # print(selected_sites)
    # # 将三组RF 横向拼接成一共数组
    # RF_merged_array = np.concatenate((RF_sites, RF_true, RF_pred), axis=1)
    # # 将RF_merged_array转为df,列名依次为Site Year Doy RF_true_NEE RF_true_GPP RF_true_RECO RF_pred_NEE RF_pred_GPP RF_pred_RECO
    # # 将RF_merged_array转换为DataFrame
    # RF_df = pd.DataFrame(RF_merged_array,
    #                      columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO", "RF_pred_NEE",
    #                               "RF_pred_GPP", "RF_pred_RECO"])
    # # 根据Site和Year进行分组
    # RF_grouped_data = RF_df.groupby(['Site', 'Year'], sort=False)
    #
    # XGB_true = XGB_RECO_sites[RF_selected_indices, 6:9]
    # XGB_pred = np.stack((XGB_NEE_predict, XGB_GPP_predict, XGB_RECO_predict), axis=1)
    # XGB_pred = XGB_pred[RF_selected_indices, :]
    # XGB_sites = XGB_RECO_sites[RF_selected_indices][:, [0, 3, 4]]
    # # 将三组RF 横向拼接成一共数组
    # XGB_merged_array = np.concatenate((XGB_sites, XGB_true, XGB_pred), axis=1)
    # # 将RF_merged_array转为df,列名依次为Site Year Doy RF_true_NEE RF_true_GPP RF_true_RECO RF_pred_NEE RF_pred_GPP RF_pred_RECO
    # # 将RF_merged_array转换为DataFrame
    # XGB_df = pd.DataFrame(XGB_merged_array,
    #                      columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO", "XGB_pred_NEE",
    #                               "XGB_pred_GPP", "XGB_pred_RECO"])
    # # 为了方便操作，将站点列表转换为 NumPy 数组
    # selected_sites_arr = np.array(selected_sites)
    # # 使用 np.where 函数查找 selected_sites 对应的索引
    # DL_selected_indices = np.where(np.isin(DL_All_true[:, :, 0], selected_sites_arr))
    # # 现在，selected_indices 包含了对应于 selected_sites 的索引值
    # # 您可以使用这些索引值来从 DL_All_true 中选择对应站点的数据
    # DL_true = DL_All_true[DL_selected_indices][:,6:9]
    # DL_pred = DL_Y_predict[DL_selected_indices]
    # DL_sites = DL_All_true[DL_selected_indices][:,[0, 3, 4]]
    #
    # DL_merged_array = np.concatenate((DL_sites, DL_true, DL_pred), axis=1)
    # DL_df = pd.DataFrame(XGB_merged_array,
    #                       columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO",
    #                                "DL_pred_NEE",
    #                                "DL_pred_GPP", "DL_pred_RECO"])
    #
    # # 使用 merge 合并三个 DataFrame
    # merged_df = DL_df.merge(XGB_df, on=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO"]).merge(RF_df,on=["Site","Year","Doy", "True_NEE","True_GPP",
    #                                                                                                            "True_RECO"])
    unique_site_ids = np.unique(np.array(RF_RECO_sites[:, 0]))

    # Randomly select sites
    np.random.seed(225)  # Set random seed
    # selected_sites = np.random.choice(unique_site_ids, 10, replace=False)
    selected_sites = ['US-Rpf', 'US-Tw1','US-NR1']
    # selected_sites = ['US-Rwe', 'US-Tw1', 'US-Ha2', 'US-A74', 'CA-ER1', 'CA-Cha', 'US-WCr', 'CA-TP4', 'US-Me6',
    #                   'US-KM3']
    # selected_sites = ['US-KM3', 'US-Rwe', 'US-Tw1','US-KM2','US-Vcm','US-Rpf','US-Wrc','US-MOz','US-NR1','US-Syv'] #
    # Obtain DL, XGBoost, and RF model data for selected sites
    RF_selected_indices = [index for index, entry in enumerate(RF_RECO_sites) if entry[0] in selected_sites]
    RF_true = RF_RECO_sites[RF_selected_indices, 6:9]
    RF_pred = np.stack((RF_NEE_predict, RF_GPP_predict, RF_RECO_predict), axis=1)[RF_selected_indices]
    RF_sites = RF_RECO_sites[RF_selected_indices][:, [0, 3, 4]]

    # Combine RF data into a single DataFrame
    RF_merged_array = np.concatenate((RF_sites, RF_true, RF_pred), axis=1)
    RF_df = pd.DataFrame(RF_merged_array,
                         columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO", "RF_pred_NEE",
                                  "RF_pred_GPP", "RF_pred_RECO"])

    # Obtain XGB data for selected sites
    XGB_true = XGB_RECO_sites[RF_selected_indices, 6:9]
    XGB_pred = np.stack((XGB_NEE_predict, XGB_GPP_predict, XGB_RECO_predict), axis=1)[RF_selected_indices]
    XGB_sites = XGB_RECO_sites[RF_selected_indices][:, [0, 3, 4]]

    # Combine XGB data into a single DataFrame
    XGB_merged_array = np.concatenate((XGB_sites, XGB_true, XGB_pred), axis=1)
    XGB_df = pd.DataFrame(XGB_merged_array,
                          columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO", "XGB_pred_NEE",
                                   "XGB_pred_GPP", "XGB_pred_RECO"])

    # Obtain DL data for selected sites
    selected_sites_arr = np.array(selected_sites)
    DL_selected_indices = np.where(np.isin(DL_All_true[:, :, 0], selected_sites_arr))
    DL_true = DL_All_true[DL_selected_indices][:, 6:9]
    DL_pred = DL_Y_predict[DL_selected_indices]
    DL_sites = DL_All_true[DL_selected_indices][:, [0, 3, 4]]

    # Combine DL data into a single DataFrame
    DL_merged_array = np.concatenate((DL_sites, DL_true, DL_pred), axis=1)
    DL_df = pd.DataFrame(DL_merged_array,
                         columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO", "DL_pred_NEE",
                                  "DL_pred_GPP", "DL_pred_RECO"])

    # Merge all three DataFrames
    merged_df = DL_df.merge(XGB_df, on=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO"]) \
        .merge(RF_df, on=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO"])

    # Print final merged DataFrame
    print(merged_df)
    print('hhh')
    # 将三个df合并

    # 将“Year”和“DOY”转换成日期
    merged_df['Date'] = pd.to_datetime(merged_df['Year'].astype(str) + '-' + merged_df['Doy'].astype(str), format='%Y-%j')

    # 站点数据按站点和变量分组
    grouped = merged_df.groupby('Site')

    # Set the font globally for the entire figure
    plt.rcParams['font.family'] = 'Times New Roman'

    # Assuming the dataframe merged_df and the grouping by 'Site' has already been performed
    # # Assuming merged_df and grouped are already defined
    # fig, axs = plt.subplots(3, 3, figsize=(8, 6))  # Adjust figsize for better visualization
    #
    # # Define sites, variables, and predictive variables
    # sites = merged_df['Site'].unique()
    # variables = ['NEE', 'GPP', 'RECO']
    # pred_variables = ['True', 'XGB_pred', 'RF_pred']  # Only using True, XGB, and RF
    # legend_variables = ['Flux tower', 'XGB', 'RF']
    #
    # # Define x-axis labels
    # years = merged_df['Year'].unique()
    # xticks = [pd.to_datetime(f'{year}-01-01') for year in years]
    #
    # # Define colors, markers, and line styles
    # colors = ['black', 'green', 'blue']
    # markers = ['o', 'o', 'o']  # Marker styles for the pred_variables
    # linestyles = ['-', '--', '-.']  # Line styles for pred_variables
    #
    # # Loop through different sites and variables and plot data
    # for i, site in enumerate(sites):
    #     site_data = grouped.get_group(site)
    #
    #     for j, var in enumerate(variables):
    #         ax = axs[i, j]
    #
    #         # Define x-axis labels for the current site based on its date range
    #         start_date = site_data['Date'].min()
    #         end_date = site_data['Date'].max()
    #         prev_year = start_date.year
    #         next_year = end_date.year + 1
    #         xticks_site = pd.date_range(start=pd.Timestamp(prev_year, 1, 1), end=pd.Timestamp(next_year, 12, 31),
    #                                     freq='YS')
    #
    #         if len(xticks_site) > 5:
    #             xticks_site = pd.date_range(start=xticks_site[0], end=xticks_site[-1],
    #                                         periods=5)  # Limit to maximum 5 labels
    #
    #         for k, pred_var in enumerate(pred_variables):
    #             marker_face_color = 'none' if pred_var == 'True' else colors[
    #                 k]  # Set marker face color to None (empty) for 'True'
    #
    #             # Only show scatter plot for the desired variables
    #             ax.scatter(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var, color=colors[k],
    #                        marker=markers[k], facecolor=marker_face_color, linewidth=1,
    #                        s=2)  # Use `s` to set point size
    #
    #         # Set title and x-axis labels for the subplot
    #         if i == 0:
    #             ax.set_title(f'{var}')
    #
    #         if j == 0:  # Add a common y-axis label for each group of three subplots in each row
    #             ax.set_ylabel(site, fontsize=12)
    #
    #         ax.set_xticks(xticks_site)
    #         ax.set_xticklabels([item.year for item in xticks_site], rotation=0)
    #
    # # Create legend and adjust layout
    # fig.text(0.025, 0.5, 'Predicted carbon fluxes (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center',
    #          rotation='vertical', fontsize=12)
    #
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.3)
    # fig.legend(legend_variables, bbox_to_anchor=(0.96, 0.07), ncol=3, fontsize=12, columnspacing=0.3)
    #
    # outpath = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    # plt.savefig(outpath + "Sites time series performance black using RLFM v4.png", format='png', dpi=300)
    # plt.show()
    # fig, axs = plt.subplots(10, 3, figsize=(10, 12))  # Adjust figsize for better visualization
    fig, axs = plt.subplots(3, 3, figsize=(8, 5.5))  # Adjust figsize for better visualization
    # Define sites, variables, and predictive variables
    sites = merged_df['Site'].unique()
    variables = ['NEE', 'GPP', 'RECO']
    pred_variables = ['True', 'XGB_pred', 'RF_pred', 'DL_pred']
    legend_variables = ['Flux tower', 'XGB', 'RF', 'Time series deep learning model with physical constraint']
    # Define x-axis labels
    years = merged_df['Year'].unique()
    xticks = [pd.to_datetime(f'{year}-01-01') for year in years]

    # Define colors, markers, and line styles
    # 黑色背景
    colors = ['black', 'green', 'blue', 'red']
    # 绿色背景
    # colors = ['#A3CB38', '#fed330', '#1e90ff', '#ff6348']
    # # 蓝色背景
    # colors = ['#34ace0', '#ffb142', '#33d9b2', '#ff5252'] #
    markers = ['o', 'o', 'o', 'o']  # Marker styles for the pred_variables
    linestyles = ['-', '--', '-.', ':']  # Line styles for pred_variables
    # Loop through different sites and variables and plot data
    for i, site in enumerate(sites):
        site_data = grouped.get_group(site)
        for j, var in enumerate(variables):
            ax = axs[i, j]

            # Define x-axis labels for the current site based on its date range
            start_date = site_data['Date'].min()
            end_date = site_data['Date'].max()
            # xticks_site = pd.date_range(start=start_date, end=end_date, freq='YS')
            prev_year = start_date.year
            next_year = end_date.year + 1
            xticks_site = pd.date_range(start=pd.Timestamp(prev_year, 1, 1), end=pd.Timestamp(next_year, 12, 31),
                                        freq='YS')

            # if len(xticks_site) > 5:
            #         xticks_site = xticks_site[[0, len(xticks_site) // 2, -1]]
            if len(xticks_site) > 5:
                xticks_site = pd.date_range(start=xticks_site[0], end=xticks_site[-1],periods=5)  # Limit to maximum 5 labels

            for k, pred_var in enumerate(pred_variables):
                # marker_style = 'o' if pred_var == 'True' else 's'  # Solid marker for 'True', empty marker for others
                marker_face_color = 'none' if pred_var == 'True' else colors[k]  # Set marker face color to None (empty) for 'True'
                # 折线+折点表示
                # ax.plot(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var, color=colors[k],
                #         marker=marker_style, markerfacecolor=marker_face_color, linestyle=linestyles[k], linewidth=1,
                #         markersize=2)
                # 只有散点
                ax.scatter(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var, color=colors[k],
                       marker=markers[k], facecolor=marker_face_color, linewidth=1, s=2)  # 使用s参数来设置点的大小
            # Set title and x-axis labels for the subplot
            if i == 0:
                ax.set_title(f'{var}')

            if j == 0:  # Add a common y-axis label for each group of three subplots in each row
                ax.set_ylabel(site,fontsize=12)

            ax.set_xticks(xticks_site)
            ax.set_xticklabels([item.year for item in xticks_site], rotation=0)

    # Create legend and adjust layout
    # fig.text(0.5, 0.06, 'Year', ha='center', va='center', fontsize=12)
    fig.text(0.025, 0.5, 'Predicted carbon fluxes (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center',
             rotation='vertical', fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.2)
    # Create legend and adjust layout
    # fig.legend(legend_variables, bbox_to_anchor=(0.955, 0.05), ncol=4, fontsize=12, columnspacing=4)
    fig.legend(legend_variables, bbox_to_anchor=(0.96, 0.07), ncol=4, fontsize=12, columnspacing=0.3)
    outpath = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    plt.savefig(outpath + "Sites time series performance black using RLFM state_90_v4.png", format='png', dpi=300)
    plt.show()

##### 每个站点为一列，一列有三行，分别代表NEE GPP RECO
    # # Create a figure with subplots
    # fig, axs = plt.subplots(3, 5, figsize=(12, 6))  # Adjust figsize for better visualization
    #
    # # Define sites, variables, and predictive variables
    # sites = merged_df['Site'].unique()
    # variables = ['NEE', 'GPP', 'RECO']
    # pred_variables = ['True', 'XGB_pred', 'RF_pred', 'DL_pred']
    # legend_variables = ['Ground observation', 'XGB_pred', 'RF_pred', 'CFR_pred']
    #
    # # Define x-axis labels
    # years = merged_df['Year'].unique()
    # xticks = [pd.to_datetime(f'{year}-01-01') for year in years]
    #
    # # Define colors, markers, and line styles
    # colors = ['black', 'green', 'blue', 'red']
    # markers = ['o', 's', '^', 'D']  # Adjust marker styles for better visibility
    # linestyles = ['-', '--', '-.', ':']  # Adjust line styles for better distinction
    #
    # # Loop through different sites and variables and plot data
    # for i, site in enumerate(sites):
    #     for j, var in enumerate(variables):
    #         ax = axs[j, i]
    #
    #         site_data = grouped.get_group(site)
    #
    #         # Define x-axis labels for the current site based on its date range
    #         start_date = site_data['Date'].min()
    #         end_date = site_data['Date'].max()
    #         xticks_site = pd.date_range(start=start_date, end=end_date, freq='YS')
    #
    #         if len(xticks_site) > 3:
    #             xticks_site = xticks_site[[0, len(xticks_site) // 2, -1]]
    #
    #         for k, pred_var in enumerate(pred_variables):
    #             marker_style = 'o' if pred_var == 'True' else 's'  # Open circle for 'True', solid marker for others
    #             ax.plot(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var, color=colors[k],
    #                     marker=markers[k], linestyle=linestyles[k], linewidth=1, markersize=2) # Adjust line width and marker size for better visibility
    #
    #         # Set title and x-axis labels for the subplot
    #         if j == 0:
    #             ax.set_title(f'{site}')
    #         ax.set_xticks(xticks_site)
    #         ax.set_xticklabels([item.year for item in xticks_site], rotation=0)  # Rotate x-axis labels if needed
    #
    # # Create legend and adjust layout
    # fig.text(0.5, 0.05, 'Year', ha='center', va='center', fontsize=12)
    # fig.text(0.03, 0.5, 'Predicted carbon fluxes (g C m$^{-2}$ d$^{-1}$)', ha='center', va='center',
    #          rotation='vertical', fontsize=12)
    # fig.text(0.06, 0.82, 'NEE', ha='center', va='center', rotation='vertical', fontsize=12)
    # fig.text(0.06, 0.5, 'GPP', ha='center', va='center', rotation='vertical', fontsize=12)
    # fig.text(0.06, 0.2, 'RECO', ha='center', va='center', rotation='vertical', fontsize=12)
    #
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.15)
    #
    # # Create legend and adjust layout
    # fig.legend(legend_variables, bbox_to_anchor=(0.94, 0.05), ncol=4, fontsize=12,columnspacing=9,frameon=False)  # Adjust legend position and ncol as needed
    # plt.show()


    # # 创建一个大图，包含三行两列的子图
    # fig, axs = plt.subplots(3, 5, figsize=(8, 4))
    #
    # # 定义站点和变量
    # sites = grouped.groups.keys()  # 获取所有站点名称
    # variables = ['NEE', 'GPP', 'RECO']
    # pred_variables = ['True', 'XGB_pred', 'RF_pred','DL_pred']
    #
    # # 定义横轴标签
    # years = merged_df['Year'].unique()
    # xticks = [pd.to_datetime(f'{year}-01-01') for year in years]
    #
    # # 定义不同预测变量的颜色和标记样式
    # colors = ['black','green', 'blue','red',]
    # markers = ['o', 'o', 'o', 'o']
    # linestyles = ['-', '-', '-', '-']
    # line_width = 0.1
    # # 循环遪列不同站点并绘制数据
    # for i, site in enumerate(sites):
    #     for j, var in enumerate(variables):
    #         ax = axs[j, i]
    #
    #         site_data = grouped.get_group(site)
    #
    #         # 获取当前站点的起止时间
    #         start_date = site_data['Date'].min()
    #         end_date = site_data['Date'].max()
    #
    #         # 定义当前站点的横轴标签
    #         xticks_site = pd.date_range(start=start_date, end=end_date, freq='YS')
    #
    #         # 如果时间序列的年份多于3个，只显示首尾和中间三个年份的横轴标签
    #         if len(xticks_site) > 3:
    #             xticks_site = xticks_site[[0, len(xticks_site) // 2, -1]]
    #
    #         for k, pred_var in enumerate(pred_variables):
    #             ax.plot(site_data['Date'], site_data[f'{pred_var}_{var}'], label=f'{pred_var}_{var}', color=colors[k],
    #                     marker=markers[k], linestyle=linestyles[k], linewidth=line_width, markersize=1)
    #
    #         # 添加标题和设置横轴标签
    #         ax.set_title(f'{site} - {var}')
    #         ax.set_xticks(xticks_site)
    #         ax.set_xticklabels([item.year for item in xticks_site], rotation=0)  # 旋转45度
    #
    # # plt.subplots_adjust(hspace=0.25, bottom=0.1)
    # # 创建共同的图例并调整位置，使其宽度与所有图相同且不覆盖图表
    # fig.legend(pred_variables,bbox_to_anchor=(0.5, 0.08),ncol=4) #  loc='lower center'
    # plt.tight_layout()
    # plt.show()

    # # 循环遍历不同站点并绘制数据
    # for i, site in enumerate(sites):
    #     for j, var in enumerate(variables):
    #         ax = axs[j, i]
    #
    #         site_data = grouped.get_group(site)
    #
    #         # 获取当前站点的起止时间
    #         start_date = site_data['Date'].min()
    #         end_date = site_data['Date'].max()
    #
    #         # 定义当前站点的横轴标签
    #         xticks_site = pd.date_range(start=start_date, end=end_date, freq='YS')
    #
    #         for k, pred_var in enumerate(pred_variables):
    #             ax.plot(site_data['Date'], site_data[f'{pred_var}_{var}'], label=f'{pred_var}_{var}', color=colors[k],
    #                     marker=markers[k], linestyle=linestyles[k], linewidth=line_width, markersize=0.5)
    #
    #         # 添加标题和设置横轴标签
    #         ax.set_title(f'{site} - {var}')
    #         ax.set_xticks(xticks_site)
    #         ax.set_xticklabels([item.year for item in xticks_site], rotation=45)  # 旋转45度
    #
    # # 调整子图布局，减小纵向间距
    # plt.tight_layout()
    #
    # # 创建共同的图例并调整位置，使其宽度与所有图相同且不覆盖图表
    # fig.legend(pred_variables, loc='lower center', ncol=4, labelspacing=0.,
    #            bbox_transform=fig.transFigure)
    #
    # plt.show()
    # # 计算行数和列数
    # nrows = (len(grouped) + 1) // 2  # 显示两列
    # ncols = 2
    # # 创建子图
    # fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), constrained_layout=True)
    # # 针对每个站点绘制图形
    # for i, (site, data) in enumerate(grouped):
    #     row = i // 2
    #     col = i % 2
    #     ax = axs[row, col]
    #     ax.set_title(f'Site: {site}')
    #     # 对数据按日期排序
    #     site_data = data.loc[data['Site'] == site].sort_values('Date')
    #     # 绘制 True_NEE、DL_pred_NEE、XGB_pred_NEE 和 RF_pred_NEE 四条折线图
    #     ax.plot(site_data['Date'], site_data['True_NEE'], label='True_NEE')
    #     ax.plot(site_data['Date'], site_data['DL_pred_NEE'], label='DL_pred_NEE')
    #     # ax.plot(site_data['Date'], site_data['XGB_pred_NEE'], label='XGB_pred_NEE')
    #     # ax.plot(site_data['Date'], site_data['RF_pred_NEE'], label='RF_pred_NEE')
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('NEE')
    #     ax.legend()
    #     # 设置横坐标刻度间隔为年份
    #     ax.xaxis.set_major_locator(mdates.YearLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # # 如果最后一行只有一个子图，需要删除多余的子图
    # if len(grouped) % 2 != 0:
    #     axs[-1, -1].remove()
    # plt.show()




