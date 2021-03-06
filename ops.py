import logging
from logging import handlers
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