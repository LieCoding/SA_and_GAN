import logging
from logging import handlers
import numpy as np
import random
import os
# 日志操作
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
    def info(self,messge):
        self.logger.info(messge)

# 划分训练集和验证集
def split_image_data(image_dir,train_rate=0.2,file_postfix='jpg',shuffle=True):
    file_name_list = os.listdir(image_dir)
    image_name_list = []
    for i in file_name_list:
        if i[-3:]==file_postfix:
            image_name_list.append(i)
            
    if shuffle==True:
        random.seed(6)
        random.shuffle(image_name_list)
    
    data_len = len(image_name_list)
    train_data = image_name_list[:int(train_rate*data_len)]
    test_data = image_name_list[int(train_rate*data_len):]
    return train_data,test_data

# CHW->HWC
def transepose_image(image,label,predict):

    t_image = []
    t_label = []
    t_predict = []

    for i,j,k in zip(label,predict,image):
        t_label.append(np.transpose(i, (1,2,0)))
        t_predict.append(np.transpose(j, (1,2,0)))
        t_image.append(np.transpose(k, (1,2,0)))

    return t_image,t_label,t_predict

#-----------get acc

import numpy as np
import cv2
import os

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""

# 从文件夹读取文件用
def flatten_data(data_dir,label_list,pre_list):
    label_data_list = []
    pre_data_list = []
    for i in range(len(label_list)):
        # 读取灰度图
        label_data = cv2.imread(os.path.join(data_dir,label_list[i]),0).astype(np.uint8)
        pre_data = cv2.imread(os.path.join(data_dir,pre_list[i]),0).astype(np.uint8)

        # 二值化，大于1的等于1
        _,label_data=cv2.threshold(label_data,1,1,cv2.THRESH_TRUNC)
        _,pre_data=cv2.threshold(pre_data,1,1,cv2.THRESH_TRUNC)

        label_data_list.append(label_data)
        pre_data_list.append(pre_data)
    label_data_list = np.array(label_data_list)
    pre_data_list =  np.array(pre_data_list)
    label_data_list = label_data_list.flatten()
    pre_data_list = pre_data_list.flatten()
    # print(label_data_list.shape)
    return label_data_list,pre_data_list

# 训练过程使用
def process_data(label_list,pre_list):
    '''
     处理数据
    :param label_list: label np格式列表
    :param pre_list:  预测图片 np格式列表
    :return: flatten的数据
    '''
    label_data_list = []
    pre_data_list = []
    for i in range(len(label_list)):
        # 二值化，大于1的等于1
        _,label_data=cv2.threshold(label_list[i].astype(np.uint8),1,1,cv2.THRESH_TRUNC)
        _,pre_data=cv2.threshold(pre_list[i].astype(np.uint8),1,1,cv2.THRESH_TRUNC)

        label_data_list.append(label_data)
        pre_data_list.append(pre_data)
    label_data_list = np.array(label_data_list)
    pre_data_list =  np.array(pre_data_list)
    label_data_list = label_data_list.flatten()
    pre_data_list = pre_data_list.flatten()
    # print(label_data_list.shape)
    return label_data_list,pre_data_list


def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength = numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return recall

def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU

def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def get_acc(label,predict,classNum=2):

    label_all,predict_all = process_data(label,predict)
    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    # OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    # FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    # mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    return precision[1],recall[1],IoU[1],f1ccore[1]

