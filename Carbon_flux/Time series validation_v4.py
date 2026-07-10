# 每个子图添加三个指标（RMSE,CC,BIAS）,每个模型都要

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
    DL_Y_predict = np.load(inputdata_path + 'DL_RLM_predict_3.npy', allow_pickle=True)
    DL_Y_true = np.load(inputdata_path + 'DL_RLM_true_3.npy', allow_pickle=True)
    DL_All_true = np.load(inputdata_path + 'DL_RLM_sites_3.npy', allow_pickle=True)
    inter_mask = np.load(inputdata_path + 'DL_RLM_x_mask_3.npy', allow_pickle=True)
    DL_y_qa = np.load(inputdata_path + 'DL_RLM_y_qa_3.npy', allow_pickle=True)
    #
    # DL_Y_predict = np.load(inputdata_path + 'DL_RLM_predict_good_modis.npy', allow_pickle=True)
    # DL_Y_true = np.load(inputdata_path + 'DL_RLM_true_good_modis.npy', allow_pickle=True)
    # DL_All_true = np.load(inputdata_path + 'DL_RLM_sites_good_modis.npy', allow_pickle=True)
    # inter_mask = np.load(inputdata_path + 'DL_RLM_x_mask_good_modis.npy', allow_pickle=True)
    # DL_y_qa = np.load(inputdata_path + 'DL_RLM_y_qa_good_modis.npy', allow_pickle=True)
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

    # 假设 DL_sites, DL_true, DL_pred, XGB_df, RF_df 已经定义并包含数据
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Combine DL data into a single DataFrame
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Combine DL data into a single DataFrame
    DL_merged_array = np.concatenate((DL_sites, DL_true, DL_pred), axis=1)
    DL_df = pd.DataFrame(DL_merged_array, columns=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO",
                                                   "DL_pred_NEE", "DL_pred_GPP", "DL_pred_RECO"])

    # Merge all three DataFrames
    merged_df = DL_df.merge(XGB_df, on=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO"]) \
        .merge(RF_df, on=["Site", "Year", "Doy", "True_NEE", "True_GPP", "True_RECO"])

    # Convert "Year" and "Doy" to date
    merged_df['Date'] = pd.to_datetime(merged_df['Year'].astype(str) + '-' + merged_df['Doy'].astype(str),
                                       format='%Y-%j')

    # Group data by site
    grouped = merged_df.groupby('Site')

    # Define sites, variables, and predictive variables
    sites = merged_df['Site'].unique()
    variables = ['NEE', 'GPP', 'RECO']
    pred_variables = [('XGB', 'XGB_pred'), ('RF', 'RF_pred'), ('Physics-aware Transformer', 'DL_pred')]  # List of tuples with model names
    legend_variables = ['Flux Tower', 'XGB', 'RF', 'Physics-aware Transformer']
    colors = ['black', 'green', 'blue', 'red']  # Ensure colors match with legend_variables

    # Prepare the plot
    plt.rcParams['font.family'] = 'Arial'
    total_plots = len(sites) * len(variables)
    fig, axs = plt.subplots(total_plots, 1, figsize=(12, 16))

    # Loop through each site and variable to create subplots
    for i, site in enumerate(sites):
        site_data = grouped.get_group(site)

        for j, var in enumerate(variables):
            ax = axs[i * len(variables) + j]  # Determine the correct subplot
            start_date = site_data['Date'].min()
            end_date = site_data['Date'].max()
            prev_year = start_date.year
            next_year = end_date.year + 1
            xticks_site = pd.date_range(start=pd.Timestamp(prev_year, 1, 1), end=pd.Timestamp(next_year, 12, 31),
                                        freq='YS')

            # Plotting true values
            ax.scatter(site_data['Date'], site_data[f'True_{var}'], label='True', color=colors[0], marker='o', s=3)

            # Initialize to collect metrics
            num_models = len(pred_variables)
            space_between_metrics = 0.1  # Adjust this value for desired spacing (total spacing will be num_models * space_between_metrics)

            for k, (model_name, pred_var) in enumerate(pred_variables):
                # Plotting each variable for predictions
                ax.scatter(site_data['Date'], site_data[pred_var + f'_{var}'], label=model_name,
                           color=colors[k + 1], marker='o', s=3)  # Adjust index to match colors correctly

                # Calculate metrics without the 'True' values
                true_values = site_data[f'True_{var}']  # True values remain for metric calculation
                predicted_values = site_data[pred_var + f'_{var}']

                # Create a mask for valid (non-NaN) entries
                valid_mask = true_values.notna() & predicted_values.notna()
                true_values_cleaned = true_values[valid_mask]
                predicted_values_cleaned = predicted_values[valid_mask]

                # Ensure the cleaned Series are not empty before computation
                if len(true_values_cleaned) > 0 and len(predicted_values_cleaned) > 0:
                    rmse = np.sqrt(np.mean((true_values_cleaned - predicted_values_cleaned) ** 2))
                    bias = np.mean(predicted_values_cleaned - true_values_cleaned)
                    cc = np.corrcoef(np.int32(true_values_cleaned), np.int32(predicted_values_cleaned))[0, 1]

                    # 计算等间距的位置
                    num_models = len(pred_variables)
                    start_x = 0.1  # 左侧起始位置（可根据需要调整）
                    end_x = 0.95  # 右侧结束位置（可根据需要调整）
                    available_space = end_x - start_x
                    space_between = available_space / (num_models - 0.05)  # 模型间的等距间隔

                    metrics_position_x = start_x + k * space_between  # 当前模型的x位置
                    metrics_position_y = 0.95  # 保持y位置不变

                    # 添加带颜色的指标文本（保持颜色和格式）
                    metrics_text = f'RMSE: {rmse:.2f} CC: {cc:.2f} Bias: {bias:.2f}'
                    ax.text(
                        metrics_position_x, metrics_position_y, metrics_text,
                        transform=ax.transAxes, fontsize=14, verticalalignment='top',
                        color=colors[k + 1], bbox=dict(facecolor='white', alpha=0.5)
                    )
            # Set title and labels
            ax.set_ylabel(f'{site} - {var}', fontsize=14)
            ax.yaxis.set_label_coords(-0.05, 0.5)
            ax.set_xticks(xticks_site)
            ax.set_xticklabels([item.year for item in xticks_site])
            ax.set_xlim(start_date, end_date)  # Set x-limits based on data

            # Set font size for x and y tick labels
            ax.tick_params(axis='both', labelsize=14)
    # Set common x-label for the last subplot
    axs[-1].set_xlabel('Year', fontsize=16)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.07, top=0.97, wspace=0.2, hspace=0.3)

    # Create a unified legend outside the plots
    fig.legend(legend_variables, bbox_to_anchor=(0.97, 0.04), ncol=4, fontsize=16, columnspacing=5, markerscale=4)

    # Saving the figure
    outpath = 'E:/The North America ecosystem carbon flux/PPT_GRAPHS/paper final figures/'
    plt.savefig(outpath + "Fig_7_0404.png", format='png', dpi=300)
    plt.show()

