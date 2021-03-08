import tensorflow as tf
import  cv2,os,time,sys
import numpy as np
import reader
import model
import ops
from tflib import conv2d

tf.reset_default_graph()

epoch =300   #总迭代次数
batch_size =32
h,w = 192, 256     # 需要输入网络的图片大小，会在读取的时候resize
a = 0.2    # 超参数,噪声所占比例
train_data_rate = 0.8    # 训练集所占比例，划分训练集和测试集

data_dir = '../dataset_new/image'    #选择数据库
#data_dir = './dataset_mini/image'    # 小样本测试程序用，共12张图片
#
data_process = True      #是否使用数据增强
log_dir = 'log/train.log'   # 设置日志保存路径


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
# 读取所有的训练集,返回tensor格式
train_data, train_label = reader.load_data(data_dir,train_images,output_size=(w, h),data_process= data_process)
# 读取所有的测试集,返回tensor格式,测试集不增强
test_data, test_label = reader.load_data(data_dir,test_images,output_size=(w, h),data_process= False)
train_log.info('------------loading success----------')


# 设置训练数据batch形式
train_input_queue = tf.train.slice_input_producer([train_data, train_label], shuffle=True)
train_image_batch, train_label_batch = tf.train.batch(train_input_queue, batch_size=batch_size, num_threads=8,
                                                                capacity=12,allow_smaller_final_batch=True)
# 设置测试数据batch形式
test_input_queue = tf.train.slice_input_producer([test_data, test_label],shuffle=False)
test_image_batch, test_label_batch = tf.train.batch(test_input_queue, batch_size=batch_size,num_threads=8,
                                                                capacity=12,allow_smaller_final_batch=True)



# 定义保存模型操作
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#这一行必须加，因为slice_input_producer的原因
    coord = tf.train.Coordinator()
    # 启动计算图中所有的队列线程
    threads = tf.train.start_queue_runners(sess,coord)

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
         # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(threads)
    #     关闭会话
        sess.close()
        train_log.info("已达到循环epoch")
        sys.exit()


     # 主线程，循环epoch
    for i in range(current_epoch,epoch):
  #       计时用
        start = time.time()

# #         循环迭代次数
        for j in range(len(train_images)//batch_size+1):

    #         由于批次类型为张量  这里先使用 run 获取到数据信息后再feed到网络中训练，
            train_feed_image,train_feed_label = sess.run([train_image_batch, train_label_batch])
            noise_z = np.random.uniform(0, 1.0, [batch_size,1,h,w])
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

#           打印当前的损失

        end = time.time()
        train_log.info('epoch:{} decoder_loss:{} D_loss:{} G_loss:{} lr:{} runing time:{} s '.format(current_epoch, _loss_decoder,_loss_D,
                                                                                    _loss_encoder2, learningrate,round((end-start),2)))

        

#         50个epoch保存一次相关图片 和模型
        if i%10 == 0 :
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_image' + '.png' , np.transpose(input_image[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_label' + '.png' , np.transpose(input_label[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_real_map' + '.png' , np.transpose( output_real_map[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_fake_map' + '.png' , np.transpose(output_fake_map[0], (1,2,0)))

            saver.save(sess, './model/fcrn.ckpt', global_step=current_epoch)

#         #1000epoch test model

        if i%50 == 0 and i!=0:
            train_log.info("--------test start---------")
            start = time.time()

            for itor in range(len(test_images)//batch_size+1):
                test_feed_data,test_feed_label = sess.run([test_image_batch, test_label_batch])
#                 print('itor ,test_feed_data,test_feed_label shape',itor,test_feed_data.shape,test_feed_label.shape)

            # 喂入网络的数据
                test_feeds = {image: test_feed_data, label: test_feed_label}
                _test_loss_decoder,_test_input_image,_test_input_label,_test_output_real_map = sess.run([ test_loss_decoder,
                                                                                                 image,label,real_map],feed_dict=test_feeds)
                for img_idx in range(len(_test_input_image)):
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" + str(img_idx)+ '_image' + '.png' , np.transpose(_test_input_image[img_idx], (1,2,0)))
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+  '_label' + '.png' , np.transpose(_test_input_label[img_idx], (1,2,0)))
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+ '_real_map' + '.png' , np.transpose(_test_output_real_map[img_idx], (1,2,0)))
                train_log.info('test save img nums:{}'.format(img_idx+1))

                train_log.info(' test:  epoch:{} itor:{} test_decoder_loss:{} '.format(current_epoch, itor,_test_loss_decoder))
            end = time.time()

            train_log.info("--------test end ,runing time:{} s ---------".format(round((end-start),2)))


#      最后保存一批次图片
    train_log.info("save final batch image!")
#     print("save final batch image!")

    for img_idx in range(len(input_image)):

        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" + str(img_idx)+ '_image' + '.png' , np.transpose(input_image[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_label' + '.png' , np.transpose(input_label[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+ '_real_map' + '.png' , np.transpose(output_real_map[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_fake_map' + '.png' , np.transpose(output_fake_map[img_idx], (1,2,0)))
    train_log.info('final save img nums:{}'.format(img_idx+1))
#     print('final save img nums:',img_idx+1)

    train_log.info("--------final_test start---------")
    start = time.time()

    for itor in range(len(test_images)//batch_size+1):
        test_feed_data,test_feed_label = sess.run([test_image_batch, test_label_batch])
#                 print('itor ,test_feed_data,test_feed_label shape',itor,test_feed_data.shape,test_feed_label.shape)

    # 喂入网络的数据
        test_feeds = {image: test_feed_data, label: test_feed_label}
        _test_loss_decoder,_test_input_image,_test_input_label,_test_output_real_map = sess.run([ test_loss_decoder,
                                                                                         image,label,real_map],feed_dict=test_feeds)
        for img_idx in range(len(_test_input_image)):
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" + str(img_idx)+ '_image' + '.png' , np.transpose(_test_input_image[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+  '_label' + '.png' , np.transpose(_test_input_label[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+ '_real_map' + '.png' , np.transpose(_test_output_real_map[img_idx], (1,2,0)))
        train_log.info('test save img nums:{}'.format(img_idx+1))

        train_log.info(' test:  epoch:{} itor:{} test_decoder_loss:{} '.format(current_epoch, itor,_test_loss_decoder))
    end = time.time()

    train_log.info("--------final test end ,runing time:{} s ---------".format(round((end-start),2)))
    train_log.info("-------------------train finish---------------------")
# #     print("Done!")

    saver.save(sess, './model/fcrn.ckpt', global_step=current_epoch)


# 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(threads)
#     关闭会话
    sess.close()


