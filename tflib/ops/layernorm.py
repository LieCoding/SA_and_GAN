import tflib as lib

import numpy as np
import tensorflow as tf


def Layernorm(name, norm_axes, inputs):
    '''
    层归一化
    :param name:
    :param norm_axes: 归一化的坐标轴，维度等等
    :param inputs: 输入
    :return:
    '''
    # 计算norm_axes维度的均值和方差，keep_dims：是否保持维度
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    # 初始化值
    offset = lib.param(name+'.offset', np.zeros(n_neurons, dtype='float32'))
    scale = lib.param(name+'.scale', np.ones(n_neurons, dtype='float32'))

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    # 变换形状，主要是为了矩阵运算而使维度相同，
    # if norm_axes = 2 --> [-1,1,1],if norm_axes = 3 --> [-1,1,1,1]
    offset = tf.reshape(offset, [-1] + [1 for i in range(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in range(len(norm_axes)-1)])

    # 批归一化，  offset 偏移量, scale 缩放大小
    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result
