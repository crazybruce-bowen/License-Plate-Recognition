# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:31:21 2022

@author: bowen
参考：https://zhuanlan.zhihu.com/p/506046289
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



#%%

def imread_photo(filename,flags = cv2.IMREAD_COLOR ):
    """
    该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
    输入： filename 指的图像文件名（可以包括路径）
          flags用来表示按照什么方式读取图片，有以下选择（默认采用彩色图像的方式）：
              IMREAD_COLOR 彩色图像
              IMREAD_GRAYSCALE 灰度图像
              IMREAD_ANYCOLOR 任意图像
    输出: 返回图片的通道矩阵
    """
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


img = imread_photo(r'D:\学习总文件夹\整体项目\车牌识别\License-Plate-Recognition\data\car_license.jpg')


def resize_photo(imgArr,MAX_WIDTH = 1000):
    """
    这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小
    输入： imgArr是输入的图像数字矩阵
    输出:  经过调整后的图像数字矩阵
    拓展：OpenCV自带的cv2.resize()函数可以实现放大与缩小，函数声明如下：
            cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
        其参数解释如下：
            src 输入图像矩阵
            dsize 二元元祖（宽，高），即输出图像的大小
            dst 输出图像矩阵
            fx 在水平方向上缩放比例，默认值为0
            fy 在垂直方向上缩放比例，默认值为0
            interpolation 插值法，如INTER_NEAREST，INTER_LINEAR，INTER_AREA，INTER_CUBIC，INTER_LANCZOS4等            
    """
    img = imgArr
    rows, cols= img.shape[:2]     #获取输入图像的高和宽
    if cols >  MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img ,( MAX_WIDTH ,int(rows * change_rate) ), interpolation = cv2.INTER_AREA)
    return img


img = resize_photo(img)


def predict(imageArr):
    """
    这个函数通过一系列的处理，找到可能是车牌的一些矩形区域
    输入： imageArr是原始图像的数字矩阵
    输出：gray_img_原始图像经过高斯平滑后的二值图
          contours是找到的多个轮廓
    """
    img_copy = imageArr.copy()
    gray_img = cv2.cvtColor(img_copy , cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5,5), 0, 0, cv2.BORDER_DEFAULT)  # 高斯卷积
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy  = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return gray_img_,contours


t = predict(img)[1]


def chose_licence_plate(contours,Min_Area = 2000):
    """
    这个函数根据车牌的一些物理特征（面积等）对所得的矩形进行过滤
    输入：contours是一个包含多个轮廓的列表，其中列表中的每一个元素是一个N*1*2的三维数组
    输出：返回经过过滤后的轮廓集合
    
    拓展：
    （1） OpenCV自带的cv2.contourArea()函数可以实现计算点集（轮廓）所围区域的面积，函数声明如下：
            contourArea(contour[, oriented]) -> retval
        其中参数解释如下：
            contour代表输入点集，此点集形式是一个n*2的二维ndarray或者n*1*2的三维ndarray
            retval 表示点集（轮廓）所围区域的面积
    （2） OpenCV自带的cv2.minAreaRect()函数可以计算出点集的最小外包旋转矩形，函数声明如下：
             minAreaRect(points) -> retval      
        其中参数解释如下：
            points表示输入的点集，如果使用的是Opencv 2.X,则输入点集有两种形式：一是N*2的二维ndarray，其数据类型只能为 int32
                                    或者float32， 即每一行代表一个点；二是N*1*2的三维ndarray，其数据类型只能为int32或者float32
            retval是一个由三个元素组成的元组，依次代表旋转矩形的中心点坐标、尺寸和旋转角度（根据中心坐标、尺寸和旋转角度
                                    可以确定一个旋转矩形）
    （3） OpenCV自带的cv2.boxPoints()函数可以根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点，函数声明如下：
             boxPoints(box[, points]) -> points
        其中参数解释如下：
            box是旋转矩形的三个属性值，通常用一个元组表示，如（（3.0，5.0），（8.0，4.0），-60）
            points是返回的四个顶点，所返回的四个顶点是4行2列、数据类型为float32的ndarray，每一行代表一个顶点坐标              
    """
    temp_contours = []
    for contour in contours:
        if cv2.contourArea( contour ) > Min_Area:
            temp_contours.append(contour)
    car_plate = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect( temp_contour )
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if 2 < aspect_ratio < 5.5:
            car_plate.append( temp_contour )
            rect_vertices = cv2.boxPoints( rect_tupple )
            rect_vertices = np.int0( rect_vertices )
    return car_plate


t2 = chose_licence_plate(t)

 
def license_segment(car_plates):
    """
    此函数根据得到的车牌定位，将车牌从原始图像中截取出来，并存在当前目录中。
    输入： car_plates是经过初步筛选之后的车牌轮廓的点集 
    输出:   "card_img.jpg"是车牌的存储名字
    """
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min,col_min = np.min(car_plate[:,0,:],axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min,col_min), (row_max, col_max), (0,255,0), 2)
            card_img = img[col_min:col_max,row_min:row_max,:]
            cv2.imshow("img", img)
        cv2.imwrite( "card_img.jpg", card_img)
        cv2.imshow("card_img.jpg", card_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return "card_img.jpg"


#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks
 
 
 
def remove_plate_upanddown_border(card_img):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    plate_Arr = cv2.imread(card_img)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold( plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    row_histogram = np.sum(plate_binary_img, axis=1)   #数组的每一行求和
    row_min = np.min( row_histogram )
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    #接下来挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1]-wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    #cv2.imshow("plate_binary_img", plate_binary_img)
 
    return  plate_binary_img
 
    ##################################################
    #测试用
    # print( row_histogram )
    # fig = plt.figure()
    # plt.hist( row_histogram )
    # plt.show()
    # 其中row_histogram是一个列表，列表当中的每一个元素是车牌二值图像每一行的灰度值之和，列表的长度等于二值图像的高度
    # 认为在高度方向，跨度最大的波峰为车牌区域
    # cv2.imshow("plate_gray_Arr", plate_binary_img[selected_wave[0]:selected_wave[1], :])
    ##################################################
    

#####################二分-K均值聚类算法############################
 
def distEclud (vecA, vecB):
    """
    计算两个坐标向量之间的街区距离 
    """
    return np.sum(abs(vecA - vecB))
 
def randCent( dataSet, k):
    n = dataSet.shape[1]  #列数
    centroids = np.zeros((k,n)) #用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:,j],axis = 0)
        rangeJ = float(np.max(dataSet[:,j])) - minJ
        for i in range(k):
            centroids[i:,j] = minJ + rangeJ * (i+1)/k
    return centroids
 
def kMeans (dataSet,k,distMeas = distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m,2))  #这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[ np.nonzero(clusterAssment[:,0]==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)
    return centroids , clusterAssment
 
 
 
def biKmeans(dataSet,k,distMeas= distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集 
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]
    clusterAssment =np.zeros((m,2))
    centroid0 = np.mean(dataSet,axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
         clusterAssment[j,1] = distMeas(np.array(centroid0),dataSet[j,:])**2
    while len(centList) <k:       #小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0] == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = np.sum(splitClustAss[:,1])
            sseNotSplit = np.sum( clusterAssment[np.nonzero(clusterAssment[:,0]!=i),1])
            if (sseSplit + sseNotSplit) < lowestSSE:         #如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0] ==1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0,:].tolist()
        centList.append( bestNewCents[1,:].tolist())
        clusterAssment[np.nonzero(clusterAssment[:,0] == bestCentTosplit)[0],:] = bestClustAss
    return centList, clusterAssment
 
 
def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array ( plate_binary_img )
    row_list,col_list = np.nonzero (  plate_binary_Arr >= 255 )
    dataArr = np.column_stack(( col_list,row_list))   #dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list =[]
    for centroids_ in  centroids_sorted:
        i = centroids.index(centroids_)
        current_class = dataArr[np.nonzero(clusterAssment[:,0]==i)[0],:]
        x_min,y_min = np.min(current_class,axis =0 )
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max,x_min,x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append( single_character_Arr )
        cv2.imshow('character'+str(i),single_character_Arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return character_list              #character_list中保存着每个字符的二值图数据
 
    ############################
    #测试用
    #print(col_histogram )
    #fig = plt.figure()
    #plt.hist( col_histogram )
    #plt.show()
    ############################


############################机器学习识别字符##########################################
#这部分是支持向量机的代码
import numpy as np
import cv2
import sklearn
 
 
def load_data(filename_1):
    """
    这个函数用来加载数据集，其中filename_1是一个文件的绝对地址
    """
    with open(filename_1, 'r') as fr_1:
        temp_address = [row.strip() for row in fr_1.readlines()]
        # print(temp_address)
        # print(len(temp_address))
    middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    sample_number = 0  # 用来计算总的样本数
    dataArr = np.zeros((13156, 400))
    label_list = []
    for i in range(len(temp_address)):
        with open(r'C:\Users\Administrator\Desktop\python code\OpenCV\121\\' + temp_address[i], 'r') as fr_2:
            temp_address_2 = [row_1.strip() for row_1 in fr_2.readlines()]
        # print(temp_address_2)
        # sample_number += len(temp_address_2)
        for j in range(len(temp_address_2)):
            sample_number += 1
            # print(middle_route[i])
            # print(temp_address_2[j])
            temp_img = cv2.imread(
                'C:\\Users\Administrator\Desktop\python code\OpenCV\plate recognition\\train\chars2\chars2\\' +
                middle_route[i] + '\\' + temp_address_2[j], cv2.COLOR_BGR2GRAY)
            # print('C:\\Users\Administrator\Desktop\python code\OpenCV\plate recognition\train\chars2\chars2\\'+ middle_route[i]+ '\\' +temp_address_2[j] )
            # cv2.imshow("temp_img",temp_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            temp_img = temp_img.reshape(1, 400)
            dataArr[sample_number - 1, :] = temp_img
        label_list.extend([i] * len(temp_address_2))
    # print(label_list)
    # print(len(label_list))
    return dataArr, np.array(label_list)
 
 
def SVM_rocognition(dataArr, label_list):
    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA
    new_dataArr = estimator.fit_transform(dataArr)
    new_testArr = estimator.fit_transform(testArr)
 
    import sklearn.svm
    svc = sklearn.svm.SVC()
    svc.fit(dataArr, label_list)  # 使用默认配置初始化SVM，对原始400维像素特征的训练数据进行建模，并在测试集上做出预测
    from sklearn.externals import joblib  # 通过joblib的dump可以将模型保存到本地，clf是训练的分类器
    joblib.dump(svc,"based_SVM_character_train_model.m")  # 保存训练好的模型，通过svc = joblib.load("based_SVM_character_train_model.m")调用
 
 
def SVM_rocognition_character( character_list ):
    character_Arr = np.zeros((len(character_list),400))
    #print(len(character_list))
    for i in range(len(character_list)):
        character_ = cv2.resize(character_list[i], (20, 20), interpolation=cv2.INTER_LINEAR)
        new_character_ = character_.reshape((1,400))[0]
        character_Arr[i,:] =  new_character_
 
    from sklearn.decomposition import PCA  # 从sklearn.decomposition 导入PCA
    estimator = PCA(n_components=20)  # 初始化一个可以将高维度特征向量（400维）压缩至20个维度的PCA
    character_Arr = estimator.fit_transform(character_Arr)
    ############
    filename_1 = r'C:\Users\Administrator\Desktop\python code\OpenCV\dizhi.txt'
    dataArr, label_list = load_data(filename_1)
    SVM_rocognition(dataArr, label_list)
    ##############
    from sklearn.externals import joblib
    clf = joblib.load("based_SVM_character_train_model.m")
    predict_result = clf.predict(character_Arr)
    middle_route = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', \
                    'G', 'H', 'J', 'K','L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    print(predict_result.tolist())
    for k in range(len(predict_result.tolist())):
        print('%c'%middle_route[predict_result.tolist()[k]])
#%%
