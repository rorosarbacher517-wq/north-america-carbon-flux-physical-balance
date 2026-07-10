import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use(['science', 'ieee'])
# plt.style.use(['science','bright'])
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.dates as mdates

if __name__ == "__main__":
    inputdata_path = './data/sites_results/'
    # linear dl
    DL_Y_predict = np.load(inputdata_path + 'DL_RL_predict_3.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RL_true_3.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RL_sites_3.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RL_x_mask_3.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RL_y_qa_3.npy', allow_pickle=True)
    DL_y_qa[:, :, 3]
    # RF three
    RF_NEE_predict = np.load(inputdata_path + 'RF_RL_NEE_predict_3.npy', allow_pickle=True)
    RF_NEE_true = np.load(inputdata_path + 'RF_RL_NEE_true_3.npy', allow_pickle=True)
    RF_NEE_sites = np.load(inputdata_path + 'RF_RL_RECO_site_3.npy', allow_pickle=True)

    RF_GPP_predict = np.load(inputdata_path + 'RF_RL_GPP_predict_3.npy', allow_pickle=True)
    RF_GPP_true = np.load(inputdata_path + 'RF_RL_GPP_true_3.npy', allow_pickle=True)
    RF_GPP_sites = np.load(inputdata_path + 'RF_RL_RECO_site_3.npy', allow_pickle=True)

    RF_RECO_predict = np.load(inputdata_path + 'RF_RL_RECO_predict_3.npy', allow_pickle=True)
    RF_RECO_true = np.load(inputdata_path + 'RF_RL_RECO_true_3.npy', allow_pickle=True)
    RF_RECO_sites = np.load(inputdata_path + 'RF_RL_RECO_site_3.npy', allow_pickle=True)
    # XGB three
    XGB_NEE_predict = np.load(inputdata_path + 'XGB_RL_NEE_predict_3.npy', allow_pickle=True)
    XGB_NEE_true = np.load(inputdata_path + 'XGB_RL_NEE_true_3.npy', allow_pickle=True)
    XGB_NEE_sites = np.load(inputdata_path + 'XGB_RL_RECO_site_3.npy', allow_pickle=True)

    XGB_GPP_predict = np.load(inputdata_path + 'XGB_RL_GPP_predict_3.npy', allow_pickle=True)
    XGB_GPP_true = np.load(inputdata_path + 'XGB_RL_GPP_true_3.npy', allow_pickle=True)
    XGB_GPP_sites = np.load(inputdata_path + 'XGB_RL_RECO_site_3.npy', allow_pickle=True)

    XGB_RECO_predict = np.load(inputdata_path + 'XGB_RL_RECO_predict_3.npy', allow_pickle=True)
    XGB_RECO_true = np.load(inputdata_path + 'XGB_RL_RECO_true_3.npy', allow_pickle=True)
    XGB_RECO_sites = np.load(inputdata_path + 'XGB_RL_RECO_site_3.npy', allow_pickle=True)

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
    # Group data by site
    grouped = merged_df.groupby('Site')
    # Define sites, variables, and predictive variables
    sites = merged_df['Site'].unique()
    variables = ['NEE', 'GPP', 'RECO']
    pred_variables = ['True', 'XGB_pred', 'RF_pred', 'DL_pred']
    legend_variables = ['Flux tower', 'XGB', 'RF', 'Physics-aware transformer']
    # colors = ['black', '#37AB78','#5EB0D2', '#D93A3B'] #'#7A7A7A'
    # colors = ['black', '#37AB78','#5EB0D2', '#D93A3B']
    colors1 = ['black', 'green', 'blue', 'red']  # Define colors for plots
    # colors2 = ['black', '#37AB78', '#589FF3', '#F94141']  # Lighter colors
    colors2 = ['#A0A0A0', '#37AB78','#8EC4D4', '#F36B6F']
    colors = []
    for c1, c2 in zip(colors1, colors2):
        rgb1 = mcolors.to_rgb(c1)
        rgb2 = mcolors.to_rgb(c2)
        new_rgb = [(r1 + r2) / 2 for r1, r2 in zip(rgb1, rgb2)]
        colors.append(mcolors.to_hex(new_rgb))
    markers = ['o', 'o', 'o', 'o']  # Marker styles for prediction variables

    plt.rcParams['font.family'] = 'Arial'

    # 所有站点的所有变量显示在一个图里
    # Create a single figure for all sites and variables
    total_plots = len(sites) * len(variables)
    fig, axs = plt.subplots(total_plots, 1, figsize=(10, 15))  # No sharing x-axis

    # Loop through each site and variable to create subplots
    for i, site in enumerate(sites):
        site_data = grouped.get_group(site)

        for j, var in enumerate(variables):
            ax = axs[i * len(variables) + j]  # Determine the correct subplot

            # # Define the unique years for the current variable
            # years = site_data['Year'].unique()
            # start_year = years.min()
            # end_year = years.max()
            # xticks_site = pd.date_range(start=pd.Timestamp(start_year, 1, 1),
            #                             end=pd.Timestamp(end_year, 12, 31),
            #                             freq='YS')  # Year start ticks
            start_date = site_data['Date'].min()
            end_date = site_data['Date'].max()
            prev_year = start_date.year
            next_year = end_date.year + 1
            xticks_site = pd.date_range(start=pd.Timestamp(prev_year, 1, 1), end=pd.Timestamp(next_year, 12, 31),freq='YS')
            # Plotting each variable
            for k, pred_var in enumerate(pred_variables):
                marker_face_color = 'none' if pred_var == 'True' else colors[k]  # Set marker face color
                ax.scatter(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var,
                           color=colors[k], marker=markers[k], facecolor=marker_face_color, s=3)

            # Set title and labels
            ax.set_ylabel(f'{site} - {var} ', fontsize=14)
            ax.set_xticks(xticks_site)
            ax.set_xticklabels([item.year for item in xticks_site])
            ax.set_xlim([site_data['Date'].min(), site_data['Date'].max()])  # Set x-limits based on data

            # Set font size for x and y tick labels
            ax.tick_params(axis='both', labelsize=14)  # Sets the font size for both x and y axis tick labels
    # Set common x-label for the last subplot
    axs[-1].set_xlabel('Year', fontsize=16)

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 0.95, 0.1])  # Leave space for the legend

    # Create a unified legend outside the plots
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Create a larger scale for the legend markers
    # for handle in by_label.values():
    #     handle.set_sizes([12])  # Increase size to 60, you can change the value as needed

    # plt.subplots_adjust(left=0.08, right=0.96, bottom=0.07, top=0.97, wspace=0.2, hspace=0.3)
    # fig.legend(legend_variables, bbox_to_anchor=(0.96, 0.04), ncol=4, fontsize=12, columnspacing=3)
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.07, top=0.97, wspace=0.2, hspace=0.3)
    fig.legend(legend_variables, bbox_to_anchor=(0.97, 0.04), ncol=4, fontsize=16, columnspacing=3,markerscale=4)  # Adjust markerscale as needed
    # fig.legend(colors, legend_variables, bbox_to_anchor=(0.96, 0.05), ncol=4, fontsize=12,
    #            columnspacing=1.2, labelspacing=1.2)  # Adjust columnspacing and labelspacing as needed
    # Saving the figure
    outpath = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(outpath + "Fig. S10225.png", format='png', dpi=300)
    plt.show()
    # 三个站点各做一个图 也就是三个图
    # Loop through each site and create subplots for the three variables
    # for i, site in enumerate(sites):
    #     site_data = grouped.get_group(site)
    #
    #     fig, axs = plt.subplots(len(variables), 1, figsize=(8, 7))  # Create a figure for each site
    #
    #     # Define x-axis labels for years
    #     years = site_data['Year'].unique()
    #     xticks_site = pd.date_range(start=pd.Timestamp(years.min(), 1, 1),
    #                                 end=pd.Timestamp(years.max(), 12, 31),
    #                                 freq='YS')  # Year start ticks
    #
    #     # Plotting each variable in its own subplot
    #     for j, var in enumerate(variables):
    #         for k, pred_var in enumerate(pred_variables):
    #             marker_face_color = 'none' if pred_var == 'True' else colors[k]  # Set marker face color
    #             axs[j].scatter(site_data['Date'], site_data[f'{pred_var}_{var}'], label=pred_var,
    #                            color=colors[k], marker=markers[k], facecolor=marker_face_color, s=1)
    #
    #         # Set title and labels
    #         if j == 0:
    #             axs[j].set_title(f'{site}', fontsize=12)
    #         axs[j].set_ylabel(f'{var} (g C m$^{{-2}}$ d$^{{-1}}$)', fontsize=12)
    #         axs[j].set_xticks(xticks_site)
    #         axs[j].set_xticklabels([item.year for item in xticks_site]) # , rotation=45
    #
    #     # Adjust layout so that legend doesn't overlap with x-axis label
    #     plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.2, hspace=0.3)
    #     fig.legend(legend_variables, bbox_to_anchor=(0.96, 0.1), ncol=4, fontsize=12, columnspacing=0.1, handletextpad=0.1)
    #
    #     # Saving the figure per site
    #     outpath = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/Graphs/model estimate results/'
    #     plt.savefig(outpath + f"Time_series_performance_{site}.png", format='png', dpi=300)
    #     plt.show()

