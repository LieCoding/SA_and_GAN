
'''
用于综合实训 标定图片

操作说明：单击左键选择点，单击左键两次后会得到两个坐标点，同时再图片中画出椭圆
        单击右键保存数据，保存标定好的图片和坐标值
        按键esc，退出程序

        不改路径的情况下，运行会自动从上一次标定的图片开始标定，通过文件夹里面的文件名来判断实现的

其他说明： color的颜色格式为RGB ,但是在标定的时候显示的是BGR，所以标定时候展示的颜色和保存的颜色会有一些区别，由于不影响标定就没有再去修改
        GetAngle_and_Length函数返回了画椭圆的必要参数，由于我标注的是船只，会出现一些比较奇怪的比例，我根据实际情况设定了特点范围下长短轴的计算方式，
        如果标定其他类别可能需要调整一下if的判断语句中的比例系数

        暂时没有撤回操作，如果标定错误请使用esc退出程序后重新标定，
        本来打算通过堆栈来写一个撤回操作，但是写的时候出现了很奇怪的撤回效果，就放弃写了，目测是我还没有理解 cv2.imshow()在窗口的显示形式

        保存到最后一张的时候会出现超出列表范围的报错，这时候已经标定完了，  也懒得写判断了

'''

import cv2
import numpy as np
from PIL import Image
import math,os
global img, im_indx,xy_out,xy_resulet


# 初始化需要标定的列表，除去已经标定的数据
def init_list(data_dir,save_dir):
    reult_list = []
    data_list = os.listdir(data_dir)
    save_list = os.listdir(save_dir)
    for i in data_list:
        if i not in save_list:
            reult_list.append(i)
    return reult_list

# 输入两个点坐标，获取线的角度值和长度
def GetAngle_and_Length(x1,y1,x2,y2):
    if (y2 - y1)<0 and (x2-x1)>0:
        arr_0 = np.array([(x2 - x1), (y1 - y2)])
        # angle += 90
    # print('(x2-x1), (y2 - y1)', (x2 - x1), (y2 - y1))
    arr_0 = np.array([(x2 - x1), (y2 - y1)])
    if (y2 - y1)<0 and (x2-x1)>0:
        arr_0 = np.array([-1* (x2 - x1), (y1 - y2)])
        # angle += 90
    elif (y2 - y1)<0 and (x2-x1)<0:
        arr_0 = np.array([-1*(x2 - x1), -1*(y1 - y2)])
    arr_1 = np.array([1, 0])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # 注意转成浮点数运算
    angle = np.arccos(cos_value) * (180/np.pi)
    length = int(math.sqrt((x2-x1)**2 + (y2 - y1)**2)*0.6)
    short_lenth = int(length * 0.4)

    # -------------根据不同的长度来设置短轴比例------------------------------------------------------------------------------
    print(length, short_lenth)
    if length>100 and length<150:

        short_lenth = int(length * 0.3)
        length = int(length * 0.8)
    elif length>150 and length<200:
        #
        short_lenth = int(length*0.2)
        length = int(length * 0.7)
    elif length>200 and length<500:
        short_lenth = int(length * 0.15)
        length = int(length*0.8)
    elif length>500:
        short_lenth = int(length * 0.14)
        length = int(length * 0.8)
    print(length, short_lenth)
    return angle,length,short_lenth


  # 保存标定图片和坐标
def save_data(img,name,xy_resulet):

    iamge_save_name = os.path.join(save_dir,'label_image/'+name)
    arr = np.array(img)
    # print(xy_resulet)
    im = Image.fromarray(arr)
    im.save(iamge_save_name)

    txt_save_name = os.path.join(save_dir,'label_txt/'+name[:-3]+'txt')
    with open(txt_save_name,'w') as f:
        for i in xy_resulet:
            for txt_id in i:
                f.write(str(txt_id)+' ')
            f.write('\n')
    print('save as ',iamge_save_name)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global img,im_indx,xy_out,xy_resulet
    # 左标定键画图
    if event == cv2.EVENT_LBUTTONDOWN :

        xy = "%d,%d" % (x, y)
        xy_out.append(xy)

        # 画线等操作
        if len(xy_out)>0 and len(xy_out)%2==0:

            # 获取x1,y1,x2,y2坐标值
            p1,p2 = xy_out[-2].split(','),xy_out[-1].split(',')
            x1,y1,x2,y2 = int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1])
            xy_resulet.append([x1,y1,x2,y2])

        #  获取中心点坐标、角度、长、短轴
            x_center,y_center = (x1+x2)//2,(y1+y2)//2
            # print( 'x_center,y_center  ',x_center,y_center)
            angel,lenth,short_lenth  = GetAngle_and_Length(x1,y1,x2,y2)
            # print('angel,lenth,short_lenth ',angel,lenth,short_lenth)
        # 画椭圆
            cv2.ellipse(img, (x_center, y_center), (lenth, short_lenth), angel, 0, 360, color, -1)  # 画椭圆


    #     右键保存数据和图片
    elif event == cv2.EVENT_RBUTTONDOWN :
        save_data(img, name_list[im_indx], xy_resulet)
        im_indx += 1
        img = cv2.imread(os.path.join(data_dir, name_list[im_indx]))
        cv2.imshow("image", img)

        xy_out = []
        xy_resulet = []  # 用于保存txt
        print('已标注:{}/{}'.format(im_indx,len(name_list)))


# -----------------------------------main------------

# 数据集路径
data_dir = 'E:/dataset/zongheshixun/dataset/ttt/image'
# 保存路径
save_dir = "E:/dataset/zongheshixun/dataset/ttt"

# 标注颜色
# color = (R, G, B)保存图片为R,G,B     使用cv2标注时候图片会显示的为B,G,R
color = (255, 0, 0)

if not os.path.exists(os.path.join(save_dir, 'label_image/')):
    os.mkdir(os.path.join(save_dir, 'label_image/'))
if not os.path.exists(os.path.join(save_dir, 'label_txt/')):
    os.mkdir(os.path.join(save_dir, 'label_txt/'))


xy_out = []
xy_resulet = []# 用于保存txt

name_list = init_list(data_dir,os.path.join(save_dir, 'label_image/'))
print('name_num',len(name_list))
im_indx = 0
# 初始图片
img = cv2.imread(os.path.join(data_dir, name_list[im_indx]))
# print (img.shape)

im0 = img

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while (True):

    try:
        cv2.imshow("image", img)
        c = cv2.waitKey(100) & 0xff

        #按esc退出
        if c==(27):
            cv2.destroyAllWindows()
            break
    except Exception:
        cv2.destroyAllWindows()
        break
