
from __future__ import division

# -*- encoding: utf-8 -*-

# ----- author info
"""
@File    : archs.py
@Time    : 2019/11/10 17:00
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description: 
This script contains all neural work architectures
'''
# ----- Load libraries 
import tensorflow as tf
import tensorlayer as tl
from src.ops import myConv2d, myConv3d, myConcat2d, myDeConv2d, myDeConv3d, myConcat3d

# ----- Custom functions 
def unet_single(x_in, is_train=False, pad='SAME'):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope("unet_single"):
        inputs = tl.layers.InputLayer(x_in, name='single_inputs')

        # encoder
        conv1 = tl.layers.Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init,
                                 name='single_conv1')

        conv2 = myConv2d(conv1, 128, pad=pad, is_train=is_train, name='single_conv2')

        conv3 = myConv2d(conv2, 256, pad=pad, is_train=is_train, name='single_conv3')

        conv4 = myConv2d(conv3, 512, pad=pad, is_train=is_train, name='single_conv4')

        conv5 = myConv2d(conv4, 512, pad=pad, is_train=is_train, name='single_conv5')

        conv6 = myConv2d(conv5, 512, pad=pad, is_train=is_train, name='single_conv6')

        conv7 = myConv2d(conv6, 512, pad=pad, is_train=is_train, name='single_conv7')

        conv8 = tl.layers.Conv2d(conv7, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init,
                                 name='single_conv8')

        # decoder
        up8 = myDeConv2d(myConcat2d(conv8, conv8), 512, pad=pad, is_train=is_train, name='single_deconv8')

        up7 = myDeConv2d(myConcat2d(up8, conv7), 512, pad=pad, is_train=is_train, name='single_deconv7')

        up6 = myDeConv2d(myConcat2d(up7, conv6), 512, pad=pad, is_train=is_train, name='single_deconv6')

        up5 = myDeConv2d(myConcat2d(up6, conv5), 512, pad=pad, is_train=is_train, name='single_deconv5')

        up4 = myDeConv2d(myConcat2d(up5, conv4), 512, pad=pad, is_train=is_train, name='single_deconv4')

        up3 = myDeConv2d(myConcat2d(up4, conv3), 512, pad=pad, is_train=is_train, name='single_deconv3')

        up2 = myDeConv2d(myConcat2d(up3, conv2), 512, pad=pad, is_train=is_train, name='single_deconv2')

        up1 = myDeConv2d(myConcat2d(up2, conv1), 512, pad=pad, is_train=is_train, name='single_deconv1')

        up0 = myConcat2d(up1, inputs)

        out = tl.layers.Conv2d(up0, 3, (4, 4), (1, 1), act=tf.nn.sigmoid, padding=pad, W_init=w_init, b_init=b_init,
                               name='single_out')
    return out


def unet_double(x_in, is_train=False, pad='SAME'):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope("unet_double"):
        inputs = tl.layers.InputLayer(x_in, name='double_inputs')

        # encoder
        conv1 = tl.layers.Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init,
                                 name='double_conv1')

        conv2 = myConv2d(conv1, 128, pad=pad, is_train=is_train, name='double_conv2')

        conv3 = myConv2d(conv2, 256, pad=pad, is_train=is_train, name='double_conv3')

        conv4 = myConv2d(conv3, 512, pad=pad, is_train=is_train, name='double_conv4')

        conv5 = myConv2d(conv4, 512, pad=pad, is_train=is_train, name='double_conv5')

        conv6 = myConv2d(conv5, 512, pad=pad, is_train=is_train, name='double_conv6')

        conv7 = myConv2d(conv6, 512, pad=pad, is_train=is_train, name='double_conv7')

        conv8 = tl.layers.Conv2d(conv7, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init,
                                 name='double_conv8')

        up8 = myDeConv3d(myConcat3d(conv8, conv8, 1), 512, strides=(2, 2, 2), pad=pad, is_train=is_train, name='double_deconv8')

        up7 = myDeConv3d(myConcat3d(up8, conv7, 2), 512, strides=(2, 2, 2), pad=pad, is_train=is_train, name='double_deconv7')

        up6 = myDeConv3d(myConcat3d(up7, conv6, 4), 512, strides=(2, 2, 2), pad=pad, is_train=is_train, name='double_deconv6')

        up5 = myDeConv3d(myConcat3d(up6, conv5, 8), 512, strides=(1, 2, 2), pad=pad, is_train=is_train, name='double_deconv5')

        up4 = myDeConv3d(myConcat3d(up5, conv4, 8), 256, strides=(1, 2, 2), pad=pad, is_train=is_train, name='double_deconv4')

        up3 = myDeConv3d(myConcat3d(up4, conv3, 8), 128, strides=(1, 2, 2), pad=pad, is_train=is_train, name='double_deconv3')

        up2 = myDeConv3d(myConcat3d(up3, conv2, 8), 64, strides=(1, 2, 2), pad=pad, is_train=is_train, name='double_deconv2')

        up1 = myDeConv3d(myConcat3d(up2, conv1, 8), 3, strides=(1, 2, 2), pad=pad, is_train=is_train, name='double_deconv1')

        out = myConcat3d(up1, inputs, 8)
    return out


def local_net(x_in, is_train=False, pad='SAME'):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope("local_net"):
        local_inputs = tl.layers.InputLayer(x_in, name='local_inputs')
        local_inputs.outputs = tf.expand_dims(local_inputs.outputs, 1)

        # local detail layers
        local_conv1 = tl.layers.DeConv3d(local_inputs, 64, filter_size=(4, 4, 4), strides=(2, 1, 1), padding=pad,
                                         W_init=w_init, b_init=b_init, name='local_conv1')

        local_conv2 = myDeConv3d(local_conv1, 64, strides=(2, 1, 1), pad=pad, is_train=is_train, name='local_conv2')

        local_conv3 = myDeConv3d(local_conv2, 64, strides=(2, 1, 1), pad=pad, is_train=is_train, name='local_conv3')

        out = myDeConv3d(local_conv3, 64, strides=(1, 1, 1), pad=pad, is_train=is_train, name='local_out')
    return out


def fusion_net(u_net_double, local_net, is_train=False, pad='SAME'):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope("fusion_net"):

        fusion_input = tl.layers.ConcatLayer([u_net_double, local_net], concat_dim=4, name='fusion_input')

        fusion_1 = myConv3d(fusion_input, 70, 32, pad=pad, is_train=is_train, name='fusion_first')

        out = tl.layers.Conv3dLayer(fusion_1, act=tf.nn.sigmoid, shape=(3, 4, 4, 32, 3), strides=(1, 1, 1, 1, 1),
                                    padding=pad, W_init=w_init, b_init=b_init, name='fusion_out')
    return out

# # FLDR + LDR -> HDR
def hybrid_net(x_in, batch_size, is_train=False, pad='SAME'):
    u_net_single_out = unet_single(x_in, is_train=is_train, pad=pad)
    merge_input = tf.concat([x_in, u_net_single_out.outputs], axis=3)
    unet_double_out = unet_double(merge_input, is_train=is_train, pad=pad)
    local_net_out = local_net(merge_input, is_train=is_train, pad=pad)
    out = fusion_net(unet_double_out, local_net_out, is_train=is_train, pad=pad)

    u_net_single_out.outputs = tf.reshape(u_net_single_out.outputs, (batch_size, -1, 3))
    out.outputs = tf.reshape(out.outputs, (batch_size, -1, 3))
    return u_net_single_out, out


# FLDR -> HDR or LDR -> HDR
def mitm_net(x_in, batch_size, is_train=False, pad='SAME'):
    unet_double_out = unet_double(x_in, is_train=is_train, pad=pad)
    local_net_out = local_net(x_in, is_train=is_train, pad=pad)
    out = fusion_net(unet_double_out, local_net_out, is_train=is_train, pad=pad)

    out.outputs = tf.reshape(out.outputs, (batch_size, -1, 3))
    return out