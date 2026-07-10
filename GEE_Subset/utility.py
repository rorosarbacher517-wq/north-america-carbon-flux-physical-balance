


import os 
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import true_color
## Plot time series
##*******************************************************************************************************************************************************
## convert patch into images
# get images 

FILL_VALUE = 65455
def get_image_fast(df_pandas, fields = ["Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2"], dtype=np.uint16):
    lon  = df_pandas['longitude'].to_numpy()
    lat  = df_pandas['latitude'] .to_numpy()
    date = df_pandas['date'] .to_numpy()
    lons  = np.sort(np.unique(lon))
    lats  = np.sort(np.unique(lat))
    dates = np.sort(np.unique(date))
    # image = np.full([lons.size, lats.size, dates.size, 4], fill_value=FILL_VALUE,dtype=dtype)
    image = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
    lon_result = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
    lat_result = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
    # image_Landsat = np.full([lons.size, lats.size, dates.size], fill_value=-9999,dtype=np.int16)
    if df_pandas.shape[0] != lons.size*lats.size*dates.size:
        print("df_pandas.shape[0] != lons.size*lats.size*dates.size:")
        return -1;
    for d in range(dates.size):
        di = date==dates[d]
        # print (df_pandas[di])
        if di.sum()!=lons.size*lats.size:
            print ('di.sum!=lons.size*lats.size at '+str(d))
            continue
        # break
        # print (fieldsi)
        ## Hank fixed this on May 06, 2024 to avoid possible errors in arrangement of matrix
        temp_lons = df_pandas['longitude'][di]
        temp_lats = df_pandas['latitude' ][di]
        for loni in range (lons.size):
            dii = np.logical_and (date==dates[d], lon==lons[loni])
            temp_lats = df_pandas['latitude' ][dii]
            sort_index = np.argsort(temp_lats)[::-1]
            for fieldsi in range(len(fields)):
                image[:,loni,d,fieldsi] = (df_pandas[fields[fieldsi]][dii].to_numpy())[sort_index]
                lon_result[:, loni, d, fieldsi] = (df_pandas['longitude'][dii].to_numpy())[sort_index]
                lat_result[:, loni, d, fieldsi] = (df_pandas['latitude'][dii].to_numpy())[sort_index]
            # image[:,:,d,fieldsi] = np.flip(df_pandas[fields[fieldsi]][di].to_numpy().reshape(lons.size, lats.size),0)
        # df_pandas['latitude'][di].to_numpy().reshape(lons.size, lats.size)
    # return image,lon_result,lat_result
    return image

## Convert to images and plot 
# def get_image_fast(df_pandas, fields = ["Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2"], dtype=np.uint16):
#     lon  = df_pandas['longitude'].to_numpy()
#     lat  = df_pandas['latitude'] .to_numpy()
#     date = df_pandas['date'] .to_numpy()
#     lons  = np.sort(np.unique(lon))
#     lats  = np.sort(np.unique(lat))
#     dates = np.sort(np.unique(date))
#     # image = np.full([lons.size, lats.size, dates.size, 4], fill_value=FILL_VALUE,dtype=dtype)
#     image = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
#     lon_result = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
#     lat_result = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
#     # image_Landsat = np.full([lons.size, lats.size, dates.size], fill_value=-9999,dtype=np.int16)
#     if df_pandas.shape[0] != lons.size*lats.size*dates.size:
#         print("df_pandas.shape[0] != lons.size*lats.size*dates.size:")
#         return -1;
#     for d in range(dates.size):
#         di = date==dates[d]
#         # print (df_pandas[di])
#         if di.sum()!=lons.size*lats.size:
#             print ('di.sum!=lons.size*lats.size at '+str(d))
#             continue
#         # break
#         # print (fieldsi)
#         ## Hank fixed this on May 06, 2024 to avoid possible errors in arrangement of matrix
#         temp_lons = df_pandas['longitude'][di]
#         temp_lats = df_pandas['latitude' ][di]
#         for loni in range (lons.size):
#             dii = np.logical_and (date==dates[d], lon==lons[loni])
#             temp_lats = df_pandas['latitude' ][dii]
#             sort_index = np.argsort(temp_lats)[::-1]
#             for fieldsi in range(len(fields)):
#                 image[:,loni,d,fieldsi] = (df_pandas[fields[fieldsi]][dii].to_numpy())[sort_index]
#                 lon_result[:, loni, d, fieldsi] = (df_pandas['longitude'][dii].to_numpy())[sort_index]
#                 lat_result[:, loni, d, fieldsi] = (df_pandas['latitude'][dii].to_numpy())[sort_index]
#             # image[:,:,d,fieldsi] = np.flip(df_pandas[fields[fieldsi]][di].to_numpy().reshape(lons.size, lats.size),0)
#         # df_pandas['latitude'][di].to_numpy().reshape(lons.size, lats.size)
#     return image,lon_result,lat_result

# def get_image_fast(df_pandas, fields = ["Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2"], dtype=np.uint16):
#     lon  = df_pandas['longitude'].to_numpy()
#     lat  = df_pandas['latitude'] .to_numpy()
#     date = df_pandas['date'] .to_numpy()
#     lons  = np.sort(np.unique(lon))
#     lats  = np.sort(np.unique(lat))
#     dates = np.sort(np.unique(date))
#     # image = np.full([lons.size, lats.size, dates.size, 4], fill_value=FILL_VALUE,dtype=dtype)
#     image = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype)
#     # image_Landsat = np.full([lons.size, lats.size, dates.size], fill_value=-9999,dtype=np.int16)
#     if df_pandas.shape[0] != lons.size*lats.size*dates.size:
#         print("df_pandas.shape[0] != lons.size*lats.size*dates.size:")
#         return -1;
#     for d in range(dates.size):
#         di = date==dates[d]
#         # print (df_pandas[di])
#         if di.sum()!=lons.size*lats.size:
#             print ('di.sum!=lons.size*lats.size at '+str(d))
#             continue
#         # break
#         # print (fieldsi)
#         ## Hank fixed this on Jun 28, 2022 to avoid possible errors in arrangement of matrix
#         temp_lons = df_pandas['longitude'][di]
#         temp_lats = df_pandas['latitude' ][di]
#         for loni in range (lons.size):
#             dii = np.logical_and (date==dates[d], lon==lons[loni])
#             temp_lats = df_pandas['latitude' ][dii]
#             sort_index = np.argsort(temp_lats)[::-1]
#             # df_pandas[fields[fieldsi]][dii]
#             # for fieldsi in range(len(fields)):
#             #     arr = df_pandas[fields[fieldsi]]
#             #     if arr is not None:
#             #         image[:, loni, d, fieldsi] = arr[sort_index]
#             #     else:
#             #         print('The data type is Nonetype')
#             for fieldsi in range(len(fields)):
#                 image[:,loni,d,fieldsi] = (df_pandas[fields[fieldsi]][dii].to_numpy())[sort_index]
#             # image[:,:,d,fieldsi] = np.flip(df_pandas[fields[fieldsi]][di].to_numpy().reshape(lons.size, lats.size),0)
#         # df_pandas['latitude'][di].to_numpy().reshape(lons.size, lats.size)
#     return image

## Convert to images and plot 
def get_image_fast_polygon(df_pandas, fields = ["Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2"], dtype=np.uint16):
    print(df_pandas)
    lon  = df_pandas['longitude'].to_numpy()
    lat  = df_pandas['latitude'] .to_numpy()
    date = df_pandas['date'] .to_numpy()
    
    lons  = np.sort(np.unique(lon))
    lats  = np.sort(np.unique(lat))
    dates = np.sort(np.unique(date))
    
    # image = np.full([lons.size, lats.size, dates.size, 4], fill_value=FILL_VALUE,dtype=dtype) 
    image = np.full([lats.size, lons.size, dates.size, len(fields)], fill_value=FILL_VALUE,dtype=dtype) 
    # image_Landsat = np.full([lons.size, lats.size, dates.size], fill_value=-9999,dtype=np.int16) 
    
    if df_pandas.shape[0] != lons.size*lats.size*dates.size:
        print("df_pandas.shape[0] != lons.size*lats.size*dates.size:")
        # return -1;
    for d in range(dates.size):
        di = date==dates[d]
        # print (df_pandas[di])
        # if di.sum()!=lons.size*lats.size:
            # print ('di.sum!=lons.size*lats.size at '+str(d))
            # continue
        
        # break
        # print (fieldsi)
        ## Hank fixed this on Jun 28, 2022 to avoid possible errors in arrangement of matrix 
        # temp_lons = df_pandas['longitude'][di]
        # temp_lats = df_pandas['latitude' ][di]
        for loni in range (lons.size):
            dii = np.logical_and (date==dates[d], lon==lons[loni])
            temp_lats = df_pandas['latitude' ][dii]
            # temp_lons = df_pandas['longitude'][dii]
            sort_index = np.argsort(temp_lats)[::-1]
            lon_sorted_index = np.searchsorted(lats, temp_lats)
            # df_pandas[fields[fieldsi]][dii]
            for fieldsi in range(len(fields)):
                image[lon_sorted_index,loni,d,fieldsi] = (df_pandas[fields[fieldsi]][dii].to_numpy())[sort_index]
            # image[:,:,d,fieldsi] = np.flip(df_pandas[fields[fieldsi]][di].to_numpy().reshape(lons.size, lats.size),0)
        
        # df_pandas['latitude'][di].to_numpy().reshape(lons.size, lats.size)
        # break
    return image

Sentinel1_dir = "./Sentinel1_images"
if not os.path.isdir(Sentinel1_dir):
	os.makedirs(Sentinel1_dir)

def plot_time_sereis (df_pandas, image, label="Landsat", figure_dir=Sentinel1_dir, offset=0, scale=1,thirdd=2):
    date = df_pandas['date'] .to_numpy()
    dates = np.sort(np.unique(date))
    dates_n = image.shape[2]
    dates_n_sqrt = int(np.ceil(np.sqrt(dates_n)))
    # create figure
    fig = plt.figure(figsize=(10, 10))
    # setting values to rows and column variables
    rows = dates_n_sqrt
    columns = dates_n_sqrt
    data = image[:,:,0,:3]
    data2 = np.full([data.shape[2], data.shape[0], data.shape[1]], fill_value=FILL_VALUE,dtype=np.int16) 
    for i in range (dates_n):
        data = image[:,:,i,:3]
        masks = data [:,:,0]!=FILL_VALUE
        data2 [0,masks] = (data [masks,0])*scale+offset
        data2 [1,masks] = (data [masks,1])*scale+offset
        data2 [2,masks] = (data [masks,thirdd])*scale+offset
        date_str = str(dates[i])[:10]
        file_name = figure_dir+'/'+label+date_str+".tif"
        # true_color.true_color_from_image (data2, file_name)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(mpimg.imread(file_name))
        plt.axis('off')
        plt.title(date_str)
        # break
    # plt.show()
    fig.show()

# df_pandas = df_pad_sentinel1
# image = image_Sentinel1
# offset=0; scale=1
# figure_dir=Sentinel1_dir
def get_time_sereis_mean_std (df_pandas, image, label="Landsat", figure_dir=Sentinel1_dir, offset=0, scale=1):
    date = df_pandas['date'] .to_numpy()
    dates_all = np.sort(np.unique(date))
    dates_n = image.shape[2]
    
    years  = [ int(str(d)[:4]  ) for d in dates_all ]
    months = [ int(str(d)[5:7] ) for d in dates_all ]
    dates  = [ int(str(d)[8:10]) for d in dates_all ]
    labels = [    (str(d)[ :10]) for d in dates_all ]
    doys = np.array([ (datetime(year=years[i], month=months[i], day=dates[i])-datetime(year=years[i], month=1, day=1)).days+1 for i in range(len(dates)) ])
    ## mean & stardard deviation 
    data = image[:,:,0,:]
    bands_n = data.shape[2]
    data2 = np.full([bands_n, data.shape[0], data.shape[1]], fill_value=FILL_VALUE,dtype=np.float) 
    means = np.full([bands_n, dates_n], fill_value=FILL_VALUE,dtype=np.float) 
    stds  = np.full([bands_n, dates_n], fill_value=FILL_VALUE,dtype=np.float) 
    for i in range (dates_n):
        data = image[:,:,i, :]
        masks = data [:,:,0]!=FILL_VALUE
        for bi in range(bands_n):
            data2 [bi,masks] = (data [masks,bi])*scale+offset
            means[bi,i] = data2 [bi,masks].mean()
            stds[bi,i]  = data2 [bi,masks].std()
        # data2 [0,masks] = (data [masks,0])*scale+offset
        # data2 [1,masks] = (data [masks,1])*scale+offset
        # data2 [2,masks] = (data [masks,thirdd])*scale+offset
        # date_str = str(dates[i])[:10]
        # file_name = figure_dir+'/'+label+date_str+".tif"
        # true_color.true_color_from_image (data2, file_name)
        # fig.add_subplot(rows, columns, i+1)
    # plt.imshow(mpimg.imread(file_name))
    # markers2 = 'o'
    # markerl8 = "^"
    
    return means, stds, doys, labels
    # plt.plot(doys, means[0,]) 
    # plt.plot(x[s2index], y[s2index], markers2, label="marker='{0}'".format(markers2), color='green',) 
    # plt.show()
    
    # plt.axis('off')
    # plt.title(label)
        # break
    
    # plt.show()
    # fig.show()

