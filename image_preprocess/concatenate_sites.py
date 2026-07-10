
import numpy as np


# 给定影像的根路径
if __name__ == "__main__":
    inputdata_path = 'E:/Paper code/code/Carbon_flux_estimation/data/sites_inputdata/'
    x_meteorological_array_flux_array1 = np.load(inputdata_path + 'y_meteorological_array_flux_0_51.npy', allow_pickle=True)
    x_meteorological_array_flux_array2 = np.load(inputdata_path + 'y_meteorological_array_flux_51_101.npy', allow_pickle=True)
    x_meteorological_array_flux_array3 = np.load(inputdata_path + 'y_meteorological_array_flux_101_201.npy', allow_pickle=True)
    x_meteorological_array_flux_array4 = np.load(inputdata_path + 'y_meteorological_array_flux_201_271.npy', allow_pickle=True)
    #  数据(波段)字段介绍见 "inputdata_path":input_data_array_introduction.xls
    x_meteorological_array_flux_array_all = np.concatenate((x_meteorological_array_flux_array1, x_meteorological_array_flux_array2, x_meteorological_array_flux_array3,x_meteorological_array_flux_array4), axis=0)
    print(x_meteorological_array_flux_array_all.shape)

    outputdata_path = 'E:/Paper code/code/Vegetation_productivity_prediction/Sites_estimation/data/Sites_images/images_match/'
    np.save(outputdata_path  + 'y_meteorological_array_flux_0_271.npy', x_meteorological_array_flux_array_all)

