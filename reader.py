
import cv2
import numpy as np
import sys ,os,random
import tensorflow as tf

def read_list(data_dir,file_postfix):
    output = []
    for i in os.listdir(data_dir):
        if i[-3:]==file_postfix:
            output.append(i)
    return output


# --------------数据增强操作--------
# 随机裁剪
def random_crop(image, label,rate=0.7):
    h, w = image.shape[:2]
    new_h = int(h*rate)
    new_w = int(w*rate)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[y:y+new_h, x:x+new_w, :]
    label = label[y:y+new_h, x:x+new_w]
    return image,label

# 随机翻转
def horizontal_flip(image,label):
    #以50%的可能性翻转图片，axis 0 垂直翻转，1水平翻转
    flip_prop = np.random.randint(low=0, high=2)
    #以50%的可能性,垂直，50水平
    axis = np.random.randint(low=0, high=2)
    if flip_prop ==0:
        image = cv2.flip(image, axis)
        label = cv2.flip(label, axis)
    return image,label

 # 添加高斯噪声
def gasuss_noise(image,label, mean=0, var=0.001):
    '''
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out,label

# 随机旋转
def rotate(image,label):
    angle = random.uniform(2,10)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    label = cv2.warpAffine(label, M, (cols, rows))
    return image,label

# 随机对比度和亮度
def Contrast_and_Brightness( image, label):

    alpha = random.uniform(1,2)
    beta = random.randint(10,100)
    blank = np.zeros(image.shape, image.dtype)
    # dst = alpha * img + beta * blank
    image = cv2.addWeighted(image, alpha, blank, 1-alpha, beta)

    return image,label

def resize_and_transpose(image,label,output_size):
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, output_size, interpolation=cv2.INTER_AREA)
    label = np.expand_dims(label,axis=-1)
    #        HWC->CHW
    image = np.transpose(image,(2,0,1))
    label = np.transpose(label,(2,0,1))
    return image,label

# ---------------------------------------



# 读取图片数据
def load_data(data_dir,image_list,output_size=(512, 384), data_process= True ):

    image_output = []
    label_output = []

    for i in range(len(image_list)):
        image_name = os.path.join(data_dir,image_list[i])
        label_name =image_name.replace('image','label')

        image = cv2.imread(image_name)
        label = cv2.imread(label_name,0)#读取灰度图

        if image is None :
            print(image_name,label_name)
            print("image无数据")
#             print(image_list)
            sys.exit()
        if label is None:
            print("label无数据")
            sys.exit()
        naive_image,naive_label = resize_and_transpose(image,label,output_size)

        image_output.append(naive_image)
        label_output.append(naive_label)

        # 使用数据增强等操作
        if data_process==True:
            crop_image,crop_label = random_crop(image,label,rate=0.7)
            crop_image,crop_label = resize_and_transpose(crop_image,crop_label,output_size)
            image_output.append(crop_image)
            label_output.append(crop_label)

            flip_image,flip_label = horizontal_flip(image,label)
            temp_image,temp_label = random_crop(flip_image,flip_label,rate=0.7)
            flip_image,flip_label = resize_and_transpose(flip_image,flip_label,output_size)
            temp_image,temp_label = resize_and_transpose(temp_image,temp_label,output_size)
            image_output.append(flip_image)
            label_output.append(flip_label)
            image_output.append(temp_image)
            label_output.append(temp_label)

            gasuss_image,gasuss_label = gasuss_noise(image,label)
            temp_image,temp_label = random_crop(gasuss_image,gasuss_label,rate=0.7)
            gasuss_image,gasuss_label = resize_and_transpose(gasuss_image,gasuss_label,output_size)
            temp_image,temp_label = resize_and_transpose(temp_image,temp_label,output_size)
            image_output.append(gasuss_image)
            label_output.append(gasuss_label)
            image_output.append(temp_image)
            label_output.append(temp_label)

            rotate_image,rotate_label = rotate(image,label)
            temp_image,temp_label = random_crop(rotate_image,rotate_label,rate=0.7)
            rotate_image,rotate_label = resize_and_transpose(rotate_image,rotate_label,output_size)
            temp_image,temp_label = resize_and_transpose(temp_image,temp_label,output_size)
            image_output.append(rotate_image)
            label_output.append(rotate_label)
            image_output.append(temp_image)
            label_output.append(temp_label)

            Bright_image,Bright_label = Contrast_and_Brightness(image,label)
            temp_image,temp_label = random_crop(Bright_image,Bright_label,rate=0.7)
            Bright_image,Bright_label = resize_and_transpose(Bright_image,Bright_label,output_size)
            temp_image,temp_label = resize_and_transpose(temp_image,temp_label,output_size)
            image_output.append(Bright_image)
            label_output.append(Bright_label)
            image_output.append(temp_image)
            label_output.append(temp_label)

    
    image_output = np.reshape(image_output, [len(image_output), 3, output_size[1], output_size[0]])#3通道
    label_output = np.reshape(label_output, [len(label_output), 1, output_size[1], output_size[0]])#单通道，若label不是单通道需修改
    # print('len(image_output)',len(image_output))
    image_output = tf.convert_to_tensor(image_output)
    label_output = tf.convert_to_tensor(label_output)
    return image_output, label_output

# 读取测试图片数据
def load_test_data(data_dir,image_list,output_size=(512, 384)):

    image_output = []
    label_output = []

    for i in range(len(image_list)):
        image_name = os.path.join(data_dir,image_list[i])
        label_name =image_name.replace('image','label')

        image = cv2.imread(image_name)
        label = cv2.imread(label_name,0)#读取灰度图

        if image is None :
            print(image_name,label_name)
            print("image无数据")
#             print(image_list)
            sys.exit()
        if label is None:
            print("label无数据")
            sys.exit()
        naive_image,naive_label = resize_and_transpose(image,label,output_size)

        image_output.append(naive_image)
        label_output.append(naive_label)

    image_output = np.reshape(image_output, [len(image_output), 3, output_size[1], output_size[0]])#3通道
    label_output = np.reshape(label_output, [len(label_output), 1, output_size[1], output_size[0]])#单通道，若label不是单通道需修改

    return image_output, label_output

