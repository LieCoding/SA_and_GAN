import tensorflow as tf
import  cv2,os,time,sys
import numpy as np
import reader
import model
import ops
from tflib import conv2d

tf.reset_default_graph()

epoch =300   #总迭代次数
batch_size =6
h,w = 192, 256     # 需要输入网络的图片大小，会在读取的时候resize
a = 0.2    # 超参数,噪声所占比例
train_data_rate = 0.8    # 训练集所占比例，划分训练集和测试集

# data_dir = './dataset_new/image'    #选择数据库
data_dir = './dataset_mini/image'    # 小样本测试程序用，共12张图片

data_process = True      #是否使用数据增强
log_dir = 'log/train.log'   # 设置日志保存路径

train_or_test = 'train'   # 训练or测试   输入：'train' or 'test'
erly_stop_epoch = 250   #早停，连续200个epoch精度没有提升就停止训练

 # 初始化log
train_log = ops.Logger(log_dir,level='debug')

# 创建记录保存的图片文件夹
if not os.path.exists('./data_save/record'):
    os.makedirs('./data_save/record')
if not os.path.exists('./data_save/final/train/'):
    os.makedirs('./data_save/final/train/')
if not os.path.exists('./data_save/final/test/'):
    os.makedirs('./data_save/final/test/')
# 保存测试数据
if not os.path.exists('./data_save/test'):
    os.makedirs('./data_save/test')



# 定义占位符
image = tf.placeholder(tf.float32, [None, 3,h, w,])
noise_image = tf.placeholder(tf.float32, [None,4,h, w])
# 单通道label
label = tf.placeholder(tf.float32, [None,1, h, w])

#获取编码器网络输出
real_feature = model.encoder1(image)
#获取噪声发生器网络encoder2的输出
fake_feature = model.encoder2(noise_image)
# 特征叠加
combine_feature = (1.0-a)*real_feature + a*fake_feature

# 真特征的检测输出
real_map  = model.decoder(real_feature)
# 混合特征的检测输出
fake_map  = model.decoder(combine_feature)

# 真特征鉴别器输出
disc_real = model.Discriminator(real_feature)
# 混合特征鉴别器输出
disc_fake = model.Discriminator(combine_feature)


# 鉴别器的损失
loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,labels=tf.ones_like(disc_real)))+\
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,labels=tf.zeros_like(disc_fake)))

# encoder2的损失函数
loss_encoder2=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))-\
        10.0*tf.reduce_mean(tf.abs(label- fake_map))

# decoder的损失函数
loss_decoder=tf.reduce_mean(tf.abs(label-real_map))+tf.reduce_mean(tf.abs(label-fake_map))

test_loss_decoder = tf.reduce_mean(tf.abs(label-real_map))



#  各网络参数
t_vars = tf.trainable_variables()
disc_vars = [var for var in t_vars if 'Discriminator' in var.name ]

encoder2_vars = [var for var in t_vars if 'encoder2' in var.name]

# 检测器，包括encoder1
decoder_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder1' in var.name]

# print(encoder2_vars)
#全局迭代次数
global_step = tf.Variable(0, trainable=False)
# 全局epoch数
global_epoch = tf.Variable(0,trainable=False, name='Parameter_name_counter')
# 定义更新epoch操作，计数用
update_epoch = tf.assign(global_epoch, tf.add(global_epoch, tf.constant(1)))


# 学习率 指数衰减衰减
lr = tf.train.exponential_decay(0.0001,global_step,2000,0.9,staircase=True)

#设置优化器
disc_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_D, var_list=disc_vars,global_step=global_step)
encoder2_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_encoder2, var_list=encoder2_vars,global_step=global_step)
decoder_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_decoder, var_list=decoder_vars,global_step=global_step)


# 获取训练数据和测试数据,返回文件名list
train_images,test_images = ops.split_image_data(data_dir,train_data_rate,file_postfix='jpg',shuffle=True)
train_log.info('-----train images:{} ,test images:{} ------'.format(len(train_images),len(test_images)))



train_log.info('------------loading train and test data-------')
# 读取所有的训练集,返回np格式
train_data, train_label = reader.load_train_data(data_dir,train_images,output_size=(w, h),data_process= data_process)

# 读取所有的测试集,返回np格式,测试集不增强
test_data, test_label = reader.load_test_data(data_dir,test_images,output_size=(w, h),)

train_log.info('------------loading success----------')


# 定义保存模型操作
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

best_acc = 0
best_epoch = 0

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    if train_or_test=='train':
#     加载检查点，方便二次训练

        train_log.info('----------------------train beginning----------------------------')
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            current_epoch = int(sess.run(global_epoch))
            train_log.info('Import models successful!  current_epoch: {} current a:{}'.format(current_epoch,a))
    #         print('Import models successful!  current_epoch:',current_epoch)
        else:
            sess.run(tf.global_variables_initializer())
            current_epoch = 0
            train_log.info('Init models successful!  current_epoch: {}  current a:{}'.format(current_epoch,a))
    #         print('Initialize successful! current_epoch:',current_epoch)

        if current_epoch>=epoch:

        #     关闭会话
            sess.close()
            train_log.info("已达到循环epoch")
            sys.exit()


         # 主线程，循环epoch
        for i in range(current_epoch,epoch):
      #       计时用
            start = time.time()

    # #         循环迭代
    #   每次epoch 都shuffle一次数据
            train_data,train_label =reader.shuffle_data(train_data,train_label)
            index = 0
            for j in range(int(len(train_images)/batch_size)+1):

                if index + batch_size >= len(train_data):
                    train_feed_image = train_data[index:]
                    train_feed_label = train_label[index:]
                else:
                    train_feed_image = train_data[index:index+batch_size]
                    train_feed_label = train_label[index:index+batch_size]


                # 当len(train_images)/batch_size余数为0时会多算一个epoch，
                if len(train_feed_image)==0:
                    break

                index+=batch_size

                noise_z = np.random.uniform(0, 1.0, [len(train_feed_image),1,h,w])
                feeed_noise_image = np.concatenate((train_feed_image,noise_z),axis=1)

        #         喂入网络的数据
                feeds = {image: train_feed_image, label: train_feed_label,noise_image:feeed_noise_image}
        #         训练网络，获取相关信息

                _loss_encoder2, _= sess.run([ loss_encoder2, encoder2_optimizer],feed_dict=feeds)
                _loss_encoder2, _= sess.run([ loss_encoder2, encoder2_optimizer],feed_dict=feeds)
                _loss_decoder, _,input_image,input_label,output_real_map,output_fake_map,learningrate = sess.run([ loss_decoder,
                                                                                                  decoder_optimizer,
                                                                                                   image,label,real_map,
                                                                                                  fake_map,lr],feed_dict=feeds)
                _loss_D, _,current_epoch= sess.run([ loss_D, disc_optimizer,global_step],feed_dict=feeds)


            current_epoch=int(sess.run(update_epoch))


    #         10个epoch保存一次相关图片
            if i%20 == 0 :
                cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_image' + '.png' , np.transpose(input_image[0], (1,2,0)))
                cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_label' + '.png' , np.transpose(input_label[0], (1,2,0)))
                cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_real_map' + '.png' , np.transpose( output_real_map[0], (1,2,0)))
                cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_fake_map' + '.png' , np.transpose(output_fake_map[0], (1,2,0)))


            # ---------每个epoch测试一次--------------
            # 初始化评价指标
            # 计算测试的迭代次数
            # 所有转置后的图片   CHW->HWC
            all_t_label = []
            all_t_predict = []
            all_t_image = []


            index = 0
            for itor in range(int(len(test_data)/batch_size)+1):

                if index + batch_size >= len(test_data):
                    test_feed_data = test_data[index:]
                    test_feed_label = test_label[index:]
                else:
                    test_feed_data = test_data[index:index+batch_size]
                    test_feed_label = test_label[index:index+batch_size]

                # 当len(test_feed_data)/batch_size余数为0时会多算一个epoch，
                if len(test_feed_data)==0:
                    break

                index+=batch_size

            # 喂入网络的数据
                test_feeds = {image: test_feed_data, label: test_feed_label}
                _test_loss_decoder,_test_input_image,_test_input_label,_test_output_real_map = sess.run([ test_loss_decoder,
                                                                                                 image,label,real_map],feed_dict=test_feeds)

                # 转置后的image , label和predict  CHW->HWC
                t_image,t_label,t_predict = ops.transepose_image(_test_input_image,_test_input_label,_test_output_real_map)

                all_t_label+=t_label
                all_t_predict+=t_predict
                all_t_image+=t_image

            # 计算精度等评价指标
            precision,recall,IoU,f1ccore = ops.get_acc(t_label,t_predict)
            # ----------------------------------------------------------------------

            # 测试集acc上升且IOU在0.6以上就保存模型
            if best_acc < precision and IoU>0.6:
                saver.save(sess, './model/fcrn.ckpt', global_step=current_epoch)
                best_acc = precision
                best_epoch = current_epoch
                train_log.info("--------update best test acc :  ---------"+ str(precision))

                # 保存最优的指标，图片信息
                _best_accs = [precision,recall,IoU,f1ccore]
                best_image,best_label,best_pre = all_t_image,all_t_label,all_t_predict

            # 早停
            if current_epoch - best_epoch > erly_stop_epoch:
                train_log.info("--------erly_stop_epoch:  ---------"+ str(current_epoch))
                break

            end = time.time()
            train_log.info('epoch:{} decoder_loss:{} D_loss:{} G_loss:{} acc:{} recall:{} IoU:{} f1ccore:{} runing time:{} s '.format(current_epoch,
                              _loss_decoder,_loss_D,_loss_encoder2, precision,recall,IoU,f1ccore,round((end-start),2)))


    #      最后保存一批次训练图片和最优的测试图片
        train_log.info("save final batch image!")
    #     print("save final batch image!")

        for img_idx in range(len(input_image)):

            cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" + str(img_idx)+ '_image' + '.png' , np.transpose(input_image[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_label' + '.png' , np.transpose(input_label[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+ '_real_map' + '.png' , np.transpose(output_real_map[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_fake_map' + '.png' , np.transpose(output_fake_map[img_idx], (1,2,0)))
        train_log.info('final save img nums:{}'.format(img_idx+1))


        # 保存精度最高模型的测试图片,避免没有更新导致best_image等不存在，使用try
        try:
            for i in range(len(best_image)):
                cv2.imwrite('./data_save/final/test/' + str(i) +  '_image' + '.png' ,best_image[i] )
                cv2.imwrite('./data_save/final/test/' + str(i) +  '_label' + '.png' ,best_label[i] )
                cv2.imwrite('./data_save/final/test/' + str(i) +  '_predict' + '.png' ,best_pre[i] )

            train_log.info("-------------------save best image in  ./data_save/final/test/ ---------------------")
            train_log.info("-------------------best epoch:"+str(best_epoch)+" bset test acc:{} recall:{} IoU:{} f1ccore:{}".format(_best_accs[0],
                                                                                    _best_accs[1],_best_accs[2],_best_accs[3]))

            train_log.info("-------------------best epoch:"+str(best_epoch)+" bset test acc:{} recall:{} IoU:{} f1ccore:{}".format(_best_accs[0],
                                                                                    _best_accs[1],_best_accs[2],_best_accs[3]))

        except:
            train_log.info("-------------------not update best data  ---------------------")

        train_log.info("-------------------train finish  ---------------------")

    elif train_or_test=='test':

        if not os.path.exists('./test_data/test/'):
            os.makedirs('./test_data/test/')

        train_log.info('----------------------test beginning----------------------------')
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            current_epoch = int(sess.run(global_epoch))
            train_log.info('Import models successful!  current_epoch: {} current a:{}'.format(current_epoch,a))
    #         print('Import models successful!  current_epoch:',current_epoch)
        else:
            print('--------no model----------')
            sys.exit()

        start = time.time()

        all_t_label = []
        all_t_predict = []
        all_t_image = []

        index = 0
        for itor in range(int(len(test_data)/batch_size)+1):

            if index + batch_size >= len(test_data):
                test_feed_data = test_data[index:]
                test_feed_label = test_label[index:]
            else:
                test_feed_data = test_data[index:index+batch_size]
                test_feed_label = test_label[index:index+batch_size]

            # 当len(test_feed_data)/batch_size余数为0时会多算一个epoch，
            if len(test_feed_data)==0:
                break
            index+=batch_size
        # 喂入网络的数据
            test_feeds = {image: test_feed_data, label: test_feed_label}
            _test_loss_decoder,_test_input_image,_test_input_label,_test_output_real_map = sess.run([ test_loss_decoder,
                                                                                             image,label,real_map],feed_dict=test_feeds)

            # 转置后的image , label和predict  CHW->HWC
            t_image,t_label,t_predict = ops.transepose_image(_test_input_image,_test_input_label,_test_output_real_map)

            all_t_label+=t_label
            all_t_predict+=t_predict
            all_t_image+=t_image

        # 计算精度等评价指标
        precision,recall,IoU,f1ccore = ops.get_acc(t_label,t_predict)

        train_log.info(' precision:{}  recall:{} IoU:{} f1ccore:{} '.format(precision,recall,IoU,f1ccore))

        # 保存图片
        for i in range(len(all_t_image)):

            cv2.imwrite('./test_data/test/' + str(i) +  '_image' + '.png' ,all_t_image[i] )
            cv2.imwrite('./test_data/test/' + str(i) +  '_label' + '.png' ,all_t_label[i] )
            cv2.imwrite('./test_data/test/' + str(i) +  '_predict' + '.png' ,all_t_predict[i] )

        train_log.info("-------------------save best image in  ./test_data/test/ ---------------------")

        end = time.time()
        train_log.info("--------test end ,runing time:{} s ---------".format(round((end-start),2)))

    else:
        print('train_or_test erro')

    #     关闭会话
        sess.close()
        train_log.info("已达到循环epoch")
        sys.exit()

#     关闭会话
    sess.close()



