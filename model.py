import tensorflow as tf
import tflib as lib
import functools
import tensorflow.contrib.slim as slim
import ops
from tflib import layernorm
from tflib import batchnorm
from tflib import conv2d
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.batchnorm.Batchnorm(name,axes,inputs,fused=True)


# def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
#     output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
#     output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
#     return output

# def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
#     output = inputs
#     output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
#     output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
#     return output
def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.nn.avg_pool(output,ksize= [1, 1, 2, 2],strides= [1, 1, 2, 2],padding= 'SAME',data_format='NCHW',name=None)
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    # print('inputs.shap:',inputs.shape)
    output = tf.nn.avg_pool(inputs,ksize= [1, 1, 2, 2],strides= [1, 1, 2, 2],padding= 'SAME',data_format='NCHW',name=None)
    # print('output.shap:',output.shape)
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def conv1x1(input_, output_dim,
              init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  # print('conv1x1 input  shape',input_.shape)
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[1], output_dim],
        initializer=init)
    # print('input_.get_shape()[1]:',input_.get_shape()[1],w)
    conv = tf.nn.conv2d(input_, w, strides=[1, 1,d_h, d_w ], padding='SAME',data_format='NCHW')
    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:

        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def self_attention1(x,k=8,name='self_attention'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        batch_size, num_channels, height, width = x.get_shape().as_list()

        # 小于k时会出现 num_channels // k = 0,调整k为通道数大小
        if num_channels < k :
            k = num_channels

        f =conv1x1(x,output_dim= num_channels//k,name='f_conv1x1')
        g =conv1x1(x,output_dim= num_channels//k,name='h_conv1x1')
        h =conv1x1(x,output_dim= num_channels,name='g_conv1x1')

        # 将f，g，h展开，方便下面的矩阵相乘
        flatten_f, flatten_g, flatten_h = tf.layers.flatten(f) ,tf.layers.flatten(g),tf.layers.flatten(h)
        s = tf.matmul(flatten_g,flatten_f,transpose_b=True)

        # attention map
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta,flatten_h)
        o = tf.reshape(o,shape=[-1, num_channels, height, width])

        sigma = tf.get_variable("sigma_ratio", [1], initializer=tf.constant_initializer(0.0))
        output = x + sigma*o

        return output
    
def self_attention(x,k=8,name='self_attention'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # BCHW
        batch_size, num_channels, height, width = x.get_shape().as_list()
        # 小于k时会出现 num_channels // k = 0,调整k为通道数大小
        if num_channels < k :
            k = num_channels
        x1 = conv1x1(x,output_dim= num_channels//k,name='x1_conv1x1')

        x1 = tf.reshape(x1,[-1,height*width,num_channels//k])
        x2 = tf.transpose(x1,(0,2,1))
        x3 = tf.matmul(x1,x2)
#         print('x1.shape,x2.shape,x3.shape',x1.shape,x2.shape,x3.shape)
        x3 = tf.nn.softmax(x3)

        x_out = tf.matmul(x3,x1)
#         print('x_out.shape',x_out.shape)
        x_out = tf.reshape(x_out,[-1,num_channels//k,height,width])
        x_out = conv1x1(x_out,output_dim= num_channels,name='xout_conv1x1')
#         print('x_out.shape',x_out.shape)
        sigma = tf.get_variable("sigma_ratio", [1], initializer=tf.constant_initializer(0.0))
        output = x + x_out*sigma
        return output

#  在残差快中增加自注意力机制
def SA_ResidualBlock(name, input_dim, output_dim, filter_size, inputs,sa=True, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)

    # 增加一层自注意力机制
    if sa==True:
        output = self_attention(output,name=name+'sa1')

    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
    # 增加一层自注意力机制
    if sa==True:
        output = self_attention(output,name=name+'sa2')

    return shortcut + output



def encoder1(inputs,dim = 8):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('encoder1')]) > 0
    with tf.variable_scope('encoder1', reuse=reuse):
        print('------------encoder1------------\n inputs shape :',inputs.shape)

        output = lib.conv2d.Conv2D('encoder1.Input', 3, dim, 3, inputs,    he_init=False)
        output = SA_ResidualBlock('encoder1.SA_Res1', dim, 2*dim, 3, output, sa=False ,  resample='down')
        output = SA_ResidualBlock('encoder1.SA_Res2', 2*dim, 4*dim, 3, output, sa=False,   resample='down')
        output = SA_ResidualBlock('encoder1.SA_Res3', 4*dim, 8*dim, 3, output,  sa=False,  resample='down')
        output = SA_ResidualBlock('encoder1.SA_Res4', 8*dim, 8*dim, 3, output,  sa=True, resample=None)

        output = Normalize('.BN1', [0,2,3], output)
        output = tf.sigmoid(output)
        print('output shape',output.shape)
        return  output


def encoder21(inputs,  dim=8, ):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('encoder2')]) > 0
    with tf.variable_scope('encoder2', reuse=reuse):
        print('------------encoder2------------\n inputs hsape :',inputs.shape)

        output = lib.conv2d.Conv2D('encoder2.Input', 4, dim, 3, inputs,    he_init=False)
        output = SA_ResidualBlock('encoder2.SA_Res1', dim, 2*dim, 3, output,  sa=False ,  resample='down')
        output = SA_ResidualBlock('encoder2.SA_Res2', 2*dim, 4*dim, 3, output, sa=False ,   resample='down')
        output = SA_ResidualBlock('encoder2.SA_Res3', 4*dim, 8*dim, 3, output,  sa=False , resample='down')
        output = SA_ResidualBlock('encoder2.SA_Res4', 8*dim, 8*dim, 3, output,  sa=True ,resample=None)

        output = Normalize('.BN1', [0,2,3], output)
        output = tf.sigmoid(output)
        print('output shape',output.shape)
        return  output
    
    
#     简化encoder2
def encoder2(inputs,  dim=8, ):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('encoder2')]) > 0
    with tf.variable_scope('encoder2', reuse=reuse):
        print('------------encoder2------------\n inputs hsape :',inputs.shape)

        output = lib.conv2d.Conv2D('encoder2.Input', 4, dim, 3, inputs,    he_init=False)
        output = tf.nn.relu(output)
        output = MeanPoolConv('.Conv1', dim, 2*dim, 3, output, he_init=True, biases=False)
        output = tf.nn.relu(output)
        output = MeanPoolConv('.Conv2', 2*dim, 4*dim, 3, output, he_init=True, biases=False)
        output = tf.nn.relu(output)
        output = MeanPoolConv('.Conv3', 4*dim, 8*dim, 3, output, he_init=True, biases=False)
        
        output = Normalize('.BN1', [0,2,3], output)
        output = tf.sigmoid(output)
        print('output shape',output.shape)
        return  output

def decoder(inputs,dim=8):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('decoder')]) > 0
    with tf.variable_scope('decoder', reuse=reuse):
        print('------------decoder------------\n input shape',inputs.shape)
        output = SA_ResidualBlock('decoder.SA_Res1', 8*dim, 8*dim, 3, inputs,  sa=True , resample='up')
        output = SA_ResidualBlock('decoder.SA_Res2', 8*dim, 4*dim, 3, output,  sa=False ,   resample='up')
        output = SA_ResidualBlock('decoder.SA_Res3', 4*dim, 2*dim, 3, output,  sa=False ,  resample='up')
        output = SA_ResidualBlock('decoder.SA_Res4', 2*dim, 1*dim, 3, output,  sa=False ,  resample=None)
        output = tf.nn.relu(output)
        de_img = lib.conv2d.Conv2D('decoder.Toimage', 1*dim, 3, 3, output)
        de_img = tf.nn.relu(de_img)
        de_img = lib.conv2d.Conv2D('decoder.Toimage2', 3, 1, 3, de_img)
        de_img = tf.nn.relu(de_img)
        print('output shape',de_img.shape)
        return  de_img

def Discriminator(inputs, dim=64 ):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Discriminator')]) > 0
    with tf.variable_scope('Discriminator', reuse=reuse):
        print('------------Discriminator------------\n inputs shappe :',inputs.shape)
        output = SA_ResidualBlock('Discriminator.SA_Res1', dim, 2*dim, 3, inputs,  sa=True ,resample='down')
        output = SA_ResidualBlock('Discriminator.SA_Res2', 2*dim, 4*dim, 3, output,  sa=False ,resample='down')
        output = SA_ResidualBlock('Discriminator.SA_Res3', 4*dim, 8*dim, 3, output, sa=False , resample='down')
        output = SA_ResidualBlock('Discriminator.SA_Res4', 8*dim, 8*dim, 3, output,  sa=False ,resample=None)
        disc_img = SA_ResidualBlock('Discriminator.SA_Res5', 8*dim, 8*dim, 3, output, sa=False, resample=None)
        print('disc_img shape',disc_img.shape)

        disc_result = tf.reduce_mean(disc_img, axis=[1,2,3],name='Discriminator.resault')
        print('output shape',disc_result.shape)
        return disc_result
