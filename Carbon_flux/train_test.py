"""
Dong adapted for LAMAP purpose
note:
    csv band refletance already in [0,1]
    to use this function, csv column order should be 23 blue band, 23 green band, 23 red band, etc.  
"""
# import train_test 
import os 
import math 
import numpy as np  
import pandas as pd 


SPLIT_DIR = "./split/"

if not os.path.isdir(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

def assign_sensor_code_xy(input_train_norm2, training_location, BANDS_N=8, SENSOR_INDEX=7,location_n=2):
    masks = input_train_norm2[:,:,:,1].copy()
    data1 = input_train_norm2[:,:,:,0].copy()
    data1[masks==0] = -9999.0
    input_train_norm3 = np.full([data1.shape[0], data1.shape[2], BANDS_N+location_n], fill_value=-9999.0, dtype=np.float32)
    input_train_norm3[:,:,:BANDS_N] = np.moveaxis(data1,1,2)  
    for ii in range(location_n):
        input_train_norm3[:,:,BANDS_N+ii] = training_location[:, (ii):(ii+1)] 
    
    sensors = [4,5,7,8]
    sensor_codes = [0, 1, 2, 3] # v8.5
    for si,sensori in enumerate(sensors):
        index_sensor = np.logical_and(input_train_norm3[:,:,SENSOR_INDEX]==sensori, input_train_norm3[:,:,SENSOR_INDEX] !=-9999)
        input_train_norm3 [index_sensor, SENSOR_INDEX] = sensor_codes[si]           
    
    return input_train_norm3

def fill_in_LST (filtered_csv_fixed_LST, data_ref):
    ##*****************************************************************************************************************
    ## fill surface temperature so that to handle when surface is not empty but ST is empty 
    if not os.path.exists(filtered_csv_fixed_LST):
        index_blue = []
        for i in range(80):
            index_blue.append('{:03d}.blue'.format(i) )
        
        index_lst = []
        for i in range(80):
            index_lst.append('{:03d}.st'.format(i) )
        
        lst_all = np.array(data_ref[index_lst ])
        lst_mean = lst_all[np.logical_and(lst_all!=0,np.logical_not (np.isnan(lst_all)))].mean() 
        data_ref['cos_as'] = np.cos(data_ref['aspect']*np.pi/180)
        data_ref['sin_as'] = np.sin(data_ref['aspect']*np.pi/180)
        # df.at[]
        # https://www.edureka.co/community/43222/python-pandas-dataframe-deprecated-removed-future-release
        for i in range (data_ref.shape[0]):
            # print (data_ref.iloc[i][index_blue])
            # print (data_ref.iloc[i][index_lst ])
            lst = np.array(data_ref.iloc[i][index_lst ]).astype(np.float32)
            indexi_blue = np.logical_not (np.isnan(np.array(data_ref.iloc[i][index_blue]).astype(np.float32)) ) 
            indexi_lst  = np.logical_and (np.logical_not (np.isnan(lst)),lst!=0) 
            indexi2 = np.logical_and (indexi_blue, lst==0) 
            # break 
            if indexi2.sum()>0:
                if indexi_lst.sum()>0:
                    # data_ref.iloc[i][index_lst[indexi2]] = lst[indexi_lst].mean() 
                    for jj in np.array(index_lst)[indexi2]:
                        data_ref.at[i,jj] = lst[indexi_lst].mean() 
                    
                else:
                    print ("data_ref.iloc[i] is empty for lst ")
                    # data_ref.iloc[i][np.array(index_lst)[indexi2]] = lst_mean
                    # data_ref[np.array(index_lst)[indexi2]][i]
                    # data_ref.set_value(i, np.array(index_lst)[indexi2], lst_mean)
                    for jj in np.array(index_lst)[indexi2]:
                        data_ref.at[i,jj] = lst_mean
                    # data_ref.at[i,np.array(index_lst)[indexi2].tolist()]
                # break 
        
        data_ref.to_csv(filtered_csv_fixed_LST) 
    else:
        data_ref = pd.read_csv(filtered_csv_fixed_LST)
    
    return data_ref

## *************************************************************************
## training and validation data split 
## invoked by customized_train.py
def random_split_train_validation (X_train,y_train,pecentage = 0.04):
    """
    split train into training and validating
    Used in the "customized_train.py" 
    """
    total_n = y_train.shape[0]
    sample_n = math.ceil(total_n*pecentage)
    split_n  = math.ceil(total_n/sample_n)
    file_index = SPLIT_DIR+"split.total_n"+str(total_n)+".for.validation.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    validation_index = orders==0
    training_index   = orders!=0
    sum(validation_index)
    sum(training_index  )
    return X_train[training_index],y_train[training_index],X_train[validation_index],y_train[validation_index],training_index,validation_index


## *************************************************************************
## training and testing data split 
## invoked by main function 
def random_split (total_n, split_n):
    """
    creat split index by split total_n into train and test based on split_n
    test: orders==0
    train: order!=0
    """
    sample_n = math.ceil(total_n/split_n)
    file_index = SPLIT_DIR+"index.total_n"+str(total_n)+".for.random.txt"
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    return orders

##***************************************************
## random split based on plot id 
# data_per = data_ref
# split_n = 10
def random_split_plotid (data_per, split_n):
    """
    creat split index by spliting plotids into 0-9 and make sure each year record has the same number as the plotid assinged number
    data_per: dataframe
	split_n = 10
    test: orders==0
    train: order!=0
    """
    unique_ids = np.sort(np.unique(data_per['plotid']))
    unique_n = 25000
    sample_n = math.ceil(unique_n/split_n)
    file_index = SPLIT_DIR+"index.plitid.total_n"+str(unique_n)+".for.random.txt"
    if os.path.isfile(file_index):
        print ('plot id order file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = np.array(dat.astype(np.int64)  )      
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        # orders = orders[range(len(unique_ids))]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")    
    
    ## order matched to id index 
    # new_orders = []
    # new_orders = np.full([data_per.shape[0]],fill_value=0,dtype=np.uint16)]
    # for ii in range(len(unique_ids)):
        # ordersi = orders[ii]
        # dataperi = data_per.loc[data_per['plotid']==unique_ids[ii]] ## unique_ids
        # new_orders[dataperi] = 
        # numi = np.repeat(ordersi, dataperi.shape[0])
        # new_orders.extend(numi)

    new_orders = np.full([data_per.shape[0]],fill_value=0,dtype=np.uint16)
    for ii, uid in enumerate(unique_ids):
        # ordersi = orders[ii]
        dataperi = data_per['plotid']==uid ## unique_ids
        new_orders[dataperi] = orders[uid-1]
        # numi = np.repeat(ordersi, dataperi.shape[0])
        # new_orders.extend(numi)
    
    return np.array(new_orders)

## ! *************************************************************************
# invoked by get_training_test_com2 function in this file 
# different from metrics - this one add masked layers to results for partial convolution to work 
# # IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
# data_all = data_per
# train_fields = train_metric
# test_field = class_field
# IMG_HEIGHT = IMG_HEIGHT2
# IMG_WIDTH = IMG_WIDTH2
# IMG_BANDS = IMG_BANDS2; is_single_norm=True; 
# construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, is_single_norm=is_single_norm)
def construct_composite_train_test(data_all,index_train,index_test,train_fields,test_field,IMG_HEIGHT,IMG_WIDTH,IMG_BANDS,mean_train=0,std_train=1, 
        is_single_norm=False,is_train_test_com=True,use_lst=False,use_ltime=False):
    """Perpare train and test from composite csv file (simple version of 'construct_metric_train_test')"""
    
    trainx2 = np.array(data_all[train_fields][index_train]).astype(np.float32)
    input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_train = np.array(data_all[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_all[train_fields][index_test]).astype(np.float32)
    input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_test = np.array(data_all[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(train_n)
    print(test_n )
    print(np.isnan(input_images_train).sum()/input_images_train.size*100)
    print(np.isnan(input_images_test ).sum()/input_images_test .size*100)
    
    ## check data
    # for i in range(input_images_train.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_train[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    # for i in range(input_images_test.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_test[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    
    ## change nan value to 0 
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0
    masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train0.copy()
    input_images_test_norm0  = input_images_test0 .copy()
    
    ## this norm turn out to be very important 
    if is_train_test_com:
        a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
            mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    else:
        a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
        
    REF_BANDS_N = 6 + use_lst + use_ltime
    if is_single_norm==True:
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=(0,2)).reshape(a.shape[1],1,1)
            std_train  = a.std (axis=(0,2)).reshape(a.shape[1],1,1)
        # mean_train = 0
        # std_train  = 1
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        ## input_images_train[0,:,:2,0]
        ## input_images_train_norm0[0,:,:2,0]
    elif is_single_norm==False:
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
            std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
        # mean_train = 0
        # std_train  = 1
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
    else: # no normalize
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=(0,2)).reshape(a.shape[1],1,1)
            std_train  = a.std (axis=(0,2)).reshape(a.shape[1],1,1)
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] 
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] 
    
    # b = np.ma.array(input_images_train_norm0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # b.mean(axis=0)
    # b.std(axis=0)
    return input_images_train_norm0,input_images_test_norm0,input_images_train0,input_images_test0,y_train,y_test,mean_train,std_train



## *************************************************************************
# invoked by main function to normalize and split data for both 16-day composite and daily
## with training and testing index returned 
# total_days=23; use_day=False; proportion=0.8; 
# total_days=80; use_day=True; proportion=0.8; is_single_norm=False; use_sensor=True; use_xy=True
# fix a bug on Dec 28, 2022 to make sure that the training and testing are the same locations 
# data_per = data_ref; IMG_HEIGHT2=TIME_SERIES_N; IMG_WIDTH2=BANDS_N;IMG_BANDS2=1; proportion=0.8; proportion2=1; total_days=TIME_SERIES_N; is_single_norm=True; use_sensor=True; use_xy=True
def get_training_test_com2_use_mean_std (data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,mean_train_pre,std_train_pre,class_field,proportion=0.1,proportion2=1,total_days=23, use_day=False, \
        is_single_norm=False, use_sensor=False, use_xy=False):
    """used construct_composite_train_test to get train and test"""
    
    ## fix training and testing split bugs 
    ## if there are many years data - make sure the training samples are from the same locations 
    if 'image_year' in data_per.keys() and 'plotid' in data_per.keys():
        print ("! fixed a bug on 2022 12 28 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        unique_years = np.sort(np.unique(data_per['image_year']))
        ref_year = unique_years[-1]
        all_plot_ids = np.array(data_per['plotid'].copy() )
        if proportion==0.8:
            ## training index using refer 2018 as divide 
            sub_index_train = np.logical_and.reduce((orders<(int)(proportion2*10),valid_index, data_per['image_year']==ref_year))
            train_plotid = data_per[sub_index_train]['plotid']
            index_train_temp = np.in1d(all_plot_ids, train_plotid)
            
            ## testing index using refer 2018 as divide 
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            
            ## note when proportion2 using refer 2018 as divide will exclude those locations which is never appear in 2018 (Jul 18, 2023) 
            if proportion2==1:  
                index_train = np.logical_not(index_test)
            else:
                index_train = np.logical_and (np.logical_not(index_test), index_train_temp )
                
        elif proportion==1.0:
            print ("proportion==1.0:")
            # sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_and(orders>-1,valid_index)
        else:
            print("shit I cannot come here")
    else:
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        if proportion==0.5:
            index_train = np.logical_and(orders<5,valid_index)
            index_test  = np.logical_and(orders>=5,valid_index)
        elif proportion==0.9:
            index_train = np.logical_and(orders<9,valid_index)
            index_test  = np.logical_and(orders>=9,valid_index)
        elif proportion==0.8:
            index_train = np.logical_and(orders<8,valid_index)
            index_test  = np.logical_and(orders>=8,valid_index)
        else:
            index_train = np.logical_and(orders==0,valid_index)
            index_test  = np.logical_and(orders!=0,valid_index)
    
    train_metric = list()
    bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2']
    if use_day:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy']
    
    if use_sensor:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy', 'sensor']
        
    for bandi in bandslist:
        # for ni in ('00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):    
        for nii in range(total_days):
            if use_day: ## this is correct as daily must use day as input 
                ni = '{:03d}'.format(nii)
            else: 
                ni = '{:02d}'.format(nii)
            
            train_metric.append(ni+'.'+bandi)
    
    
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, \
        mean_train=mean_train_pre,std_train=std_train_pre, is_single_norm=is_single_norm, is_train_test_com=proportion!=1.0)
    
    if use_xy:
        location_index = ['x','y','dem','slope']
        # mean_x = np.array(data_per[['x','y']].mean())
        # std_x  = np.array(data_per[['x','y']].std ())
        # mean_train = np.concatenate((mean_train, mean_x.reshape(2,1,1) ) )
        # std_train  = np.concatenate((std_train , std_x .reshape(2,1,1) ) )
        mean_x = np.ma.getdata(mean_train_pre)[:,0,0] [-4:]
        
        std_x  =  np.ma.getdata(std_train_pre)[:,0,0] [-4:]
        
        training_location = (np.array(data_per[index_train][location_index].copy())-mean_x)/std_x
        testing_location  = (np.array(data_per[index_test ][location_index].copy())-mean_x)/std_x        
    
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    # dat_out['predicted_rf1' ] = 255
    # dat_out['predicted_rf2' ] = 255
    
    if use_xy:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location
    else:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test



## *************************************************************************
# ! proportion is used for training/testing split; can only be set as 80% and 100%, respectively, by Sep 25, 2023 
# ! proportion2 is used to further reduce training after fixing (1-proportion) as training; can be any values <proportion, by Sep 25, 2023 

# invoked by main function to normalize and split data for both 16-day composite and daily
## with training and testing index returned 
# total_days=23; use_day=False; proportion=0.8; 
# total_days=80; use_day=True; proportion=0.8; is_single_norm=False; use_sensor=True; use_xy=True
# fix a bug on Dec 28, 2022 to make sure that the training and testing are the same locations 
# (data_pre,orders_pre,orders_pre>-1,TIME_SERIES_N_pre,BANDS_N,IMG_BANDS2,class_field,proportion=1.0, total_days=TIME_SERIES_N_pre, \
        # use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
# data_per = data_ref                # data_per = data_pre                
# orders = orders                    # orders = orders_pre             
# valid_index = valid_index          # valid_index = orders_pre>-1     
# IMG_HEIGHT2 = TIME_SERIES_N        # IMG_HEIGHT2 = TIME_SERIES_N_pre
# IMG_WIDTH2 = BANDS_N               # IMG_WIDTH2 = BANDS_N            
# proportion=0.8                     # proportion=1                      
# total_days=TIME_SERIES_N           # total_days=TIME_SERIES_N_pre    
# use_day=True                       # use_day=True                        
def get_training_test_com_lst(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=0.1, proportion2=1.0, total_days=23, use_day=False, \
        is_single_norm=False, use_sensor=False, use_xy=False, use_lst=False, use_ltime=False):
    """used construct_composite_train_test to get train and test"""
    
    ## fix training and testing split bugs 
    ## if there are many years data - make sure the training samples are from the same locations 
    if 'image_year' in data_per.keys() and 'plotid' in data_per.keys():
        print ("! fixed a bug on 2022 12 28 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        unique_years = np.sort(np.unique(data_per['image_year']))
        ref_year = unique_years[-1]
        all_plot_ids = np.array(data_per['plotid'].copy() )
        if proportion==0.8:
            ## training index using refer 2018 as divide using proportion2
            sub_index_train = np.logical_and.reduce((orders<(int)(proportion2*10),valid_index, data_per['image_year']==ref_year))
            train_plotid = data_per[sub_index_train]['plotid']
            index_train_temp = np.in1d(all_plot_ids, train_plotid)
            
            ## testing index using refer 2018 as divide 
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            
            ## note when proportion2 using refer 2018 as divide will exclude those locations which is never appear in 2018 (Jul 18, 2023) 
            if proportion2==1:  
                index_train = np.logical_not(index_test)
            else:
                index_train = np.logical_and (np.logical_not(index_test), index_train_temp )
        
        elif proportion==1.0:
            print ("proportion==1.0:")
            # sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_and(orders>-1,valid_index)
        else:
            print("shit I cannot come here")
    else:
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        if proportion==0.5:
            index_train = np.logical_and(orders<5 ,valid_index)
            index_test  = np.logical_and(orders>=5,valid_index)
        elif proportion==0.9:
            index_train = np.logical_and(orders<9 ,valid_index)
            index_test  = np.logical_and(orders>=9,valid_index)
        elif proportion==0.8:
            index_train = np.logical_and(orders<8,valid_index)
            index_test  = np.logical_and(orders>=8,valid_index)
        else:
            index_train = np.logical_and(orders==0,valid_index)
            index_test  = np.logical_and(orders!=0,valid_index)
    
    train_metric = list()
    bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2']
    if use_day:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy']
    
    if use_sensor:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy', 'sensor']

    if use_sensor and use_lst:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'doy', 'sensor']
        
    if use_sensor and use_lst and use_ltime:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'ltime', 'doy', 'sensor']        
    
    for bandi in bandslist:
        # for ni in ('00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):    
        for nii in range(total_days):
            if use_day: ## this is correct as daily must use day as input 
                ni = '{:03d}'.format(nii)
            else: 
                ni = '{:02d}'.format(nii)
            
            train_metric.append(ni+'.'+bandi)

    
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, is_single_norm=is_single_norm, is_train_test_com=proportion!=1.0, use_lst=use_lst, use_ltime=use_ltime)
    
    
    if use_xy:
        # location_index = ['x','y','dem','slope','cos_as','sin_as']
        location_index = ['x','y','dem','slope'] # deleted sin and cosine on Jul 14, 2023 
        mean_x = np.array(data_per[location_index].mean())
        std_x  = np.array(data_per[location_index].std ())
        mean_train = np.concatenate((mean_train, mean_x.reshape(mean_x.size,1,1) ) )
        std_train  = np.concatenate((std_train , std_x .reshape(mean_x.size,1,1) ) )
        training_location = (np.array(data_per[index_train][location_index].copy())-mean_x)/std_x
        testing_location  = (np.array(data_per[index_test ][location_index].copy())-mean_x)/std_x        
    
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    
    if use_xy:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location
    else:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test



## *************************************************************************
# Oct 26, 2023 
# ! proportion is used for training/testing split; can only be set as 80% and 100%, respectively, by Sep 25, 2023 
# ! proportion2 is used to further reduce training after fixing (1-proportion) as training; can be any values <proportion, by Sep 25, 2023 

# invoked by main function to normalize and split data for both 16-day composite and daily
## with training and testing index returned 
# total_days=23; use_day=False; proportion=0.8; 
# total_days=80; use_day=True; proportion=0.8; is_single_norm=False; use_sensor=True; use_xy=True
# fix a bug on Dec 28, 2022 to make sure that the training and testing are the same locations 
# (data_pre,orders_pre,orders_pre>-1,TIME_SERIES_N_pre,BANDS_N,IMG_BANDS2,class_field,proportion=1.0, total_days=TIME_SERIES_N_pre, \
        # use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
# data_per = data_ref                # data_per = data_pre                
# orders = orders                    # orders = orders_pre             
# valid_index = valid_index          # valid_index = orders_pre>-1     
# IMG_HEIGHT2 = TIME_SERIES_N        # IMG_HEIGHT2 = TIME_SERIES_N_pre
# IMG_WIDTH2 = BANDS_N               # IMG_WIDTH2 = BANDS_N            
# proportion=0.8                     # proportion=1                      
# total_days=TIME_SERIES_N           # total_days=TIME_SERIES_N_pre    
# use_day=True                       # use_day=True                        
def get_training_test_com_lst_plotID(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=0.1, proportion2=1.0, total_days=23, use_day=False, \
        is_single_norm=False, use_sensor=False, use_xy=False, use_lst=False, use_ltime=False, use_year=False,foldi=-1):
    """used construct_composite_train_test to get train and test"""
    if proportion<1.0:
        print ("! fixed a bug on 2023 10 26 to use orders ")
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        if foldi==-1:
            index_test = np.logical_and(orders>(int)(proportion*10),valid_index)
        else:
            index_test = np.logical_and.reduce((orders>foldi*2,orders<=(foldi*2+2),valid_index)) # Feb 25. 2024 
        
        ## note when proportion2 using refer 2018 as divide will exclude those locations which is never appear in 2018 (Jul 18, 2023) 
        if proportion2==1:  
            index_train = np.logical_not(index_test)
        else:
            index_train = np.logical_and(orders<=(int)(proportion2*10),valid_index)
        
    else:
        index_test = np.logical_and(orders>8,valid_index)
        index_train = valid_index
    
    train_metric = list()
    bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2']
    if use_day:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy']
    
    if use_sensor:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy', 'sensor']

    if use_sensor and use_lst:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'doy', 'sensor']
        
    if use_sensor and use_lst and use_ltime:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'ltime', 'doy', 'sensor']        
    
    for bandi in bandslist:
        # for ni in ('00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):    
        for nii in range(total_days):
            if use_day: ## this is correct as daily must use day as input 
                ni = '{:03d}'.format(nii)
            else: 
                ni = '{:02d}'.format(nii)
            
            train_metric.append(ni+'.'+bandi)

    
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, is_single_norm=is_single_norm, is_train_test_com=proportion!=1.0, use_lst=use_lst, use_ltime=use_ltime)
    
    
    if use_xy:
        # location_index = ['x','y','dem','slope','cos_as','sin_as']
        location_index = ['x','y','dem','slope'] # deleted sin and cosine on Jul 14, 2023 
        mean_x = np.array(data_per[location_index].mean())
        std_x  = np.array(data_per[location_index].std ())
        mean_train = np.concatenate((mean_train, mean_x.reshape(mean_x.size,1,1) ) )
        std_train  = np.concatenate((std_train , std_x .reshape(mean_x.size,1,1) ) )
        training_location = (np.array(data_per[index_train][location_index].copy())-mean_x)/std_x
        testing_location  = (np.array(data_per[index_test ][location_index].copy())-mean_x)/std_x        
    
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    image_year_list = ['image_year','plotid']
    training_year,testing_year = data_per[image_year_list][index_train], data_per[image_year_list][index_test]
    if use_xy:
        if use_year:
            return y_train,y_test,\
                input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location,training_year,testing_year
        else:
            return y_train,y_test,\
                input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location
    else:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test




## *************************************************************************
# Oct 27, 2023 - testing 
# ! proportion is used for training/testing split; can only be set as 80% and 100%, respectively, by Sep 25, 2023 
# ! proportion2 is used to further reduce training after fixing (1-proportion) as training; can be any values <proportion, by Sep 25, 2023 

# invoked by main function to normalize and split data for both 16-day composite and daily
## with training and testing index returned 
# total_days=23; use_day=False; proportion=0.8; 
# total_days=80; use_day=True; proportion=0.8; is_single_norm=False; use_sensor=True; use_xy=True
# fix a bug on Dec 28, 2022 to make sure that the training and testing are the same locations 
# (data_pre,orders_pre,orders_pre>-1,TIME_SERIES_N_pre,BANDS_N,IMG_BANDS2,class_field,proportion=1.0, total_days=TIME_SERIES_N_pre, \
        # use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
# data_per = data_ref                # data_per = data_pre                
# orders = orders                    # orders = orders_pre             
# valid_index = valid_index          # valid_index = orders_pre>-1     
# IMG_HEIGHT2 = TIME_SERIES_N        # IMG_HEIGHT2 = TIME_SERIES_N_pre
# IMG_WIDTH2 = BANDS_N               # IMG_WIDTH2 = BANDS_N            
# proportion=0.8                     # proportion=1                      
# total_days=TIME_SERIES_N           # total_days=TIME_SERIES_N_pre    
# use_day=True                       # use_day=True                        
def get_training_test_com_lst_plotID_test3year(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=0.1, proportion2=1.0, total_days=23, use_day=False, \
        is_single_norm=False, use_sensor=False, use_xy=False, use_lst=False, use_ltime=False):
    """used construct_composite_train_test to get train and test"""
    
    if proportion<1.0:
        print ("! fixed a bug on 2023 10 26 to use orders ")
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        index_test = np.logical_and(orders>(int)(proportion*10),valid_index)
        ## note when proportion2 using refer 2018 as divide will exclude those locations which is never appear in 2018 (Jul 18, 2023) 
        if proportion2==1:  
            # index_train = np.logical_not(index_test)
            index_train = np.logical_and(np.logical_not(index_test),np.logical_or.reduce((data_per['image_year']==1985, data_per['image_year']==2006, data_per['image_year']==2018)) )
        else:
            # index_train = np.logical_and(orders<=(int)(proportion2*10),valid_index)
            index_train = np.logical_and(orders<=(int)(proportion2*10),valid_index,np.logical_or.reduce((data_per['image_year']==1985, data_per['image_year']==2006, data_per['image_year']==2018)) )
        
    else:
        index_test = np.logical_and(orders>8,valid_index)
        index_train = valid_index
    
    train_metric = list()
    bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2']
    if use_day:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy']
    
    if use_sensor:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy', 'sensor']

    if use_sensor and use_lst:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'doy', 'sensor']
        
    if use_sensor and use_lst and use_ltime:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'ltime', 'doy', 'sensor']        
    
    for bandi in bandslist:
        # for ni in ('00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):    
        for nii in range(total_days):
            if use_day: ## this is correct as daily must use day as input 
                ni = '{:03d}'.format(nii)
            else: 
                ni = '{:02d}'.format(nii)
            
            train_metric.append(ni+'.'+bandi)

    
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, is_single_norm=is_single_norm, is_train_test_com=proportion!=1.0, use_lst=use_lst, use_ltime=use_ltime)
    
    
    if use_xy:
        # location_index = ['x','y','dem','slope','cos_as','sin_as']
        location_index = ['x','y','dem','slope'] # deleted sin and cosine on Jul 14, 2023 
        mean_x = np.array(data_per[location_index].mean())
        std_x  = np.array(data_per[location_index].std ())
        mean_train = np.concatenate((mean_train, mean_x.reshape(mean_x.size,1,1) ) )
        std_train  = np.concatenate((std_train , std_x .reshape(mean_x.size,1,1) ) )
        training_location = (np.array(data_per[index_train][location_index].copy())-mean_x)/std_x
        testing_location  = (np.array(data_per[index_test ][location_index].copy())-mean_x)/std_x        
    
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    
    if use_xy:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location
    else:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test




train_fields1 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    # for peri in ('low10', 'low25', 'middle','high75', 'high90'):
    for peri in ('low25', 'middle','high75'):
        # print(bandi)
        train_fields1.append(peri+'.'+bandi)

train_fields5 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low25', 'middle','high75', 'high90'):
        train_fields5.append(peri+'.'+bandi)

train_fields7 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low20', 'low35', 'middle','high65', 'high80', 'high90'):
        train_fields7.append(peri+'.'+bandi)
        
train_fields9 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low20', 'low30', 'low40', 'middle','high60', 'high70', 'high80', 'high90'):
        train_fields9.append(peri+'.'+bandi)

train_fields_com = list()
for comi in range(14):
    peri='N{:02d}'.format(comi)
    for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi'):
        train_fields_com.append(peri+'.'+bandi)


## *************************************************************************
# invoked by get_training_test_metric function in this file 
# IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
# IMG_HEIGHT = 5   ; IMG_WIDTH=14; IMG_BANDS=1
def construct_metric_train_test(data_per,index_train,index_test,train_fields,test_field,IMG_HEIGHT,IMG_WIDTH,IMG_BANDS=0):        
    """
    function: read csv data (filled empty with 0 or "interpolation value") and perpare standardized train and test data for model
    data_per: csv array by filter out un-used information
    index_train: split index for train
    index_test:  split index for test
    train_fileds: just use bands related columns 
    IMG_HEIGHT: metric
    IMG_WIDTH:  band number
    return: 
        csv>>image (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS) NOTE: have to be this shape, otherwise the value will shift 
        train and test of x
        normalized train and test of x
        train and test of y
        mean_train,std_train
    """
    trainx2 = np.array(data_per[train_fields][index_train]).astype(np.float32)
    trainx2[np.isnan(trainx2)] = 0 
    trainx2[np.isinf(trainx2)] = 0 
    if IMG_BANDS==0:
        input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT)    
    else:
        input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    
    y_train = np.array(data_per[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_per[train_fields][index_test]).astype(np.float32)
    testx2[np.isnan(testx2)] = 0 
    testx2[np.isinf(testx2)] = 0 
    if IMG_BANDS==0:
        input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT)
    else:
        input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    
    y_test = np.array(data_per[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(f"total train number: {train_n}")
    print(f"total test number:  {test_n}" )
    print(np.isnan(input_images_train).sum()/input_images_train.size*100)
    print(np.isnan(input_images_test ).sum()/input_images_test .size*100)
    
    ## check data
    # for i in range(input_images_train.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_train[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    # for i in range(input_images_test.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_test[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    
    # input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    # masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    # input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    # input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0       # set all nan = 0 
    # masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    # input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    # input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train.copy()
    input_images_test_norm0  = input_images_test .copy()
    
    ## this norm turn out to be very important 
    # a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
        # mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    
    if IMG_BANDS==0:
        a = np.concatenate((input_images_train[:,:,:],input_images_test[:,:,:]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2])
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2])
        input_images_train_norm0[:,:,:] = (input_images_train[:,:,:] - mean_train)/std_train
        input_images_test_norm0 [:,:,:] = (input_images_test [:,:,:] - mean_train)/std_train
    else:
        a = np.concatenate((input_images_train[:,:,:,0],input_images_test[:,:,:,0]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
        input_images_train_norm0[:,:,:,:IMG_BANDS] = (input_images_train[:,:,:,:IMG_BANDS] - mean_train)/std_train
        input_images_test_norm0 [:,:,:,:IMG_BANDS] = (input_images_test [:,:,:,:IMG_BANDS] - mean_train)/std_train
    # b = np.ma.array(input_images_train_norm0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # b.mean(axis=0)
    # b.std(axis=0)
    return input_images_train_norm0,input_images_test_norm0,input_images_train,input_images_test,y_train,y_test,mean_train,std_train


## *************************************************************************
# invoked by main function to normalize and split data for metrics percentiles 
def get_training_test_metric(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.1):
    ## fix training and testing split bugs 
    ## if there are many years data - make sure the training samples are from the same locations 
    if 'image_year' in data_per.keys() and 'plotid' in data_per.keys():
        print ("! fixed a bug on 2022 12 28 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        unique_years = np.sort(np.unique(data_per['image_year']))
        ref_year = unique_years[-1]
        all_plot_ids = np.array(data_per['plotid'].copy() )
        if proportion==0.8:
            sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_not(index_test)  
        elif proportion==1.0:
            print ("proportion==1.0:")
            # sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_and(orders>-1,valid_index)
        else:
            print("shit I cannot come here")
    else:
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        if proportion==0.5:
            index_train = np.logical_and(orders<5,valid_index)
            index_test  = np.logical_and(orders>=5,valid_index)
        elif proportion==0.9:
            index_train = np.logical_and(orders<9,valid_index)
            index_test  = np.logical_and(orders>=9,valid_index)
        elif proportion==0.8:
            index_train = np.logical_and(orders<8,valid_index)
            index_test  = np.logical_and(orders>=8,valid_index)
        else:
            index_train = np.logical_and(orders==0,valid_index)
            index_test  = np.logical_and(orders!=0,valid_index)
    
    # IMG_HEIGHT2 = 3   ; IMG_WIDTH2=13; IMG_BANDS2=1
    train_fieldsx = train_fields7
    if IMG_HEIGHT2 == 7:
        train_fieldsx = train_fields7
    elif IMG_HEIGHT2 == 9:
        train_fieldsx = train_fields9
    elif IMG_HEIGHT2 == 5:
        train_fieldsx = train_fields5
    elif IMG_HEIGHT2 == 3:
        train_fieldsx = train_fields1
    else:
        train_fieldsx = train_fields_com
        
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train2,std_train2 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2)
    
    # for 1d cnn
    input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,y_train,y_test,mean_train3,std_train3 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2)
    
    dat_out = pd.DataFrame()
    # for ai in ('hh', 'vv', 'col', 'row', 'nlcd'):
    for ai in ('x', 'y', 'image_year', 'dominant_landuse'):
        dat_out[ai] = (data_per[ai][index_test]).copy()
    
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    dat_out['predicted_rf1' ] = 255
    dat_out['predicted_rf2' ] = 255
    return y_train,y_test,\
        input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,\
        input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,dat_out, mean_train2,std_train2,index_train,index_test       

