import gdal, osr
import numpy as np
import os


#  数组保存为tif
def array2raster(TifName, GeoTransform, array):
    cols = array.shape[1]  # 矩阵列数
    rows = array.shape[0]  # 矩阵行数
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(TifName, cols, rows, 1, gdal.GDT_Float32)
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform(tuple(GeoTransform))
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    # 代码4326表示WGS84坐标
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


#  hdf批量转tif
def hdf2tif_batch(hdfFolder):
    #  获取文件夹内的文件名
    hdfNameList = os.listdir(hdfFolder)
    for i in range(len(hdfNameList)):
        #  判断当前文件是否为HDF文件
        if (os.path.splitext(hdfNameList[i])[1] == ".hdf"):
            hdfPath = hdfFolder + "/" + hdfNameList[i]
            #  gdal打开hdf数据集
            datasets = gdal.Open(hdfPath)

            #  获取hdf中的元数据
            Metadata = datasets.GetMetadata()
            #  获取四个角的维度
            Latitudes = Metadata["GRINGPOINTLATITUDE.1"]
            #  采用", "进行分割
            LatitudesList = Latitudes.split(", ")
            #  获取四个角的经度
            Longitude = Metadata["GRINGPOINTLONGITUDE.1"]
            #  采用", "进行分割
            LongitudeList = Longitude.split(", ")

            # 图像四个角的地理坐标
            GeoCoordinates = np.zeros((4, 2), dtype="float32")
            GeoCoordinates[0] = np.array([float(LongitudeList[0]), float(LatitudesList[0])])
            GeoCoordinates[1] = np.array([float(LongitudeList[1]), float(LatitudesList[1])])
            GeoCoordinates[2] = np.array([float(LongitudeList[2]), float(LatitudesList[2])])
            GeoCoordinates[3] = np.array([float(LongitudeList[3]), float(LatitudesList[3])])

            #  列数
            Columns = float(Metadata["DATACOLUMNS"])
            #  行数
            Rows = float(Metadata["DATAROWS"])
            #  图像四个角的图像坐标
            PixelCoordinates = np.array([[0, 0],
                                         [Columns - 1, 0],
                                         [Columns - 1, Rows - 1],
                                         [0, Rows - 1]], dtype="float32")

            #  计算仿射变换矩阵
            from scipy.optimize import leastsq
            def func(i):
                Transform0, Transform1, Transform2, Transform3, Transform4, Transform5 = i[0], i[1], i[2], i[3], i[4], \
                i[5]
                return [Transform0 + PixelCoordinates[0][0] * Transform1 + PixelCoordinates[0][1] * Transform2 -
                        GeoCoordinates[0][0],
                        Transform3 + PixelCoordinates[0][0] * Transform4 + PixelCoordinates[0][1] * Transform5 -
                        GeoCoordinates[0][1],
                        Transform0 + PixelCoordinates[1][0] * Transform1 + PixelCoordinates[1][1] * Transform2 -
                        GeoCoordinates[1][0],
                        Transform3 + PixelCoordinates[1][0] * Transform4 + PixelCoordinates[1][1] * Transform5 -
                        GeoCoordinates[1][1],
                        Transform0 + PixelCoordinates[2][0] * Transform1 + PixelCoordinates[2][1] * Transform2 -
                        GeoCoordinates[2][0],
                        Transform3 + PixelCoordinates[2][0] * Transform4 + PixelCoordinates[2][1] * Transform5 -
                        GeoCoordinates[2][1],
                        Transform0 + PixelCoordinates[3][0] * Transform1 + PixelCoordinates[3][1] * Transform2 -
                        GeoCoordinates[3][0],
                        Transform3 + PixelCoordinates[3][0] * Transform4 + PixelCoordinates[3][1] * Transform5 -
                        GeoCoordinates[3][1]]

            #  最小二乘法求解
            GeoTransform = leastsq(func, np.asarray((1, 1, 1, 1, 1, 1)))

            #  获取数据时间
            date = Metadata["RANGEBEGINNINGDATE"]

            #  第一个子数据集合,也就是NDVI数据
            DatasetNDVI = datasets.GetSubDatasets()[0][0]
            RasterNDVI = gdal.Open(DatasetNDVI)
            NDVI = RasterNDVI.ReadAsArray()

            TifName = date + ".tif"
            array2raster(TifName, GeoTransform[0], NDVI)
            print(TifName, "Saved successfully!")


hdf2tif_batch(r"E:/Carbon_flux/data/MODIS Land Cover")