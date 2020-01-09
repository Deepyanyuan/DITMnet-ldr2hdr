from __future__ import division

# -*- encoding: utf-8 -*-
"""
@File    : ops.py
@Time    : 2019/11/11 9:47
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description:
This script contains all neural network layers and functions that are used the project.
'''
# ----- Load libraries -------------------------------------
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import glob, math
from src import img_io

# ----- Custom functions -----
weight_init = tf.contrib.layers.xavier_initializer()

def myConv2d(last_layer, out_filter, pad, is_train, name):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net = tl.layers.Conv2d(last_layer, out_filter, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init,
                                 name=name + '_front')
    net = tl.layers.BatchNormLayer(net, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=gamma_init, name=name)
    return net

def myConv3d(last_layer, in_filter, out_filter, pad, is_train, name):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net = tl.layers.Conv3dLayer(last_layer, act=None, shape=(3, 4, 4, in_filter, out_filter), strides=(1, 1, 1, 1, 1),
                                     padding=pad, W_init=w_init, b_init=b_init)
    net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                                        name=name)
    return net


def myConcat2d(last_layer, front_layer):
    net = tl.layers.ConcatLayer([last_layer, front_layer], concat_dim=3)
    return net

def myDeConv2d(last_layer, out_filter, pad, is_train, name):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net = tl.layers.DeConv2d(last_layer, out_filter, filter_size=(4, 4), strides=(2, 2), padding=pad, W_init=w_init,
                             b_init=b_init)
    net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name=name)
    return net


def myDeConv3d(last_layer, out_filter, strides, pad, is_train, name):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net = tl.layers.DeConv3d(last_layer, out_filter, filter_size=(4, 4, 4), strides=strides, padding=pad, W_init=w_init,
                            b_init=b_init)
    net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                                  name=name)
    return net


def myConcat3d(last_layer, front_layer, scale):
    front_layer.outputs = tf.expand_dims(front_layer.outputs, 1)
    front_layer = tl.layers.TileLayer(front_layer, [1, scale, 1, 1, 1])
    net = tl.layers.ConcatLayer([last_layer, front_layer], concat_dim=4)
    return net


def myCost3(FLAGS, y_out, y_gt, norm_y_out, norm_y_gt, x_out, x_gt):
    # define cost function
    cost1 = FLAGS.cost1_weight * tl.cost.absolute_difference_error(y_out, y_gt, is_mean=True,
                                                              name='absolute_difference_error_loss')
    cost2 = FLAGS.cost2_weight * tf.losses.cosine_distance(norm_y_out, norm_y_gt, axis=2)
    cost3 = FLAGS.cost3_weight * tl.cost.absolute_difference_error(x_out, x_gt, is_mean=True,
                                                              name='absolute_difference_error_loss')
    cost = cost1 + cost2 + cost3
    return cost1, cost2, cost3, cost


def myCost2(FLAGS, y_out, y_gt, norm_y_out, norm_y_gt):
    # define cost function
    cost1 = FLAGS.cost1_weight * tl.cost.absolute_difference_error(y_out, y_gt, is_mean=True,
                                                              name='absolute_difference_error_loss')
    cost2 = FLAGS.cost2_weight * tf.losses.cosine_distance(norm_y_out, norm_y_gt, axis=2)
    cost = cost1 + cost2
    return cost1, cost2, cost


def myLossMask(FLAGS):
    # create a mask
    lossmask_list = list()
    img_shape = (FLAGS.height, FLAGS.width, 3)
    for i in range(FLAGS.predicted_window_len):
        lossmask = np.ones(img_shape[0] * img_shape[1] * img_shape[2]).reshape((1,) + img_shape[:])
        for j in range(FLAGS.predicted_window_len - 1, 0, -1):
            if i < j:
                append_img = np.ones(img_shape[0] * img_shape[1] * img_shape[2]).reshape((1,) + img_shape[:])
            else:
                append_img = np.zeros(img_shape[0] * img_shape[1] * img_shape[2]).reshape((1,) + img_shape[:])
            lossmask = np.vstack([lossmask, append_img])

        lossmask = np.broadcast_to(lossmask, (FLAGS.batch_size,) + lossmask.shape).astype(np.float32)
        lossmask_list.append(lossmask)
    lossmask_list = np.array(lossmask_list)
    return lossmask_list


def mySummary(FLAGS, cost1, cost2, cost3, cost):
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('cost1', cost1)
    tf.summary.scalar('cost2', cost2)
    tf.summary.scalar('cost3', cost3)
    tf.summary.scalar('lr', FLAGS.lr)
    summary_op = tf.summary.merge_all()
    return summary_op


def myPerFile(FLAGS, dir_path_list, perm, i):
    is_upexposure_trained = int(FLAGS.learn_type)
    # test code
    dir_path = dir_path_list[perm[i]]
    img_path_list_H = glob.glob(dir_path + '/HDR/1.hdr')  # 对数域且归一化处理
    img_path_list_F = glob.glob(dir_path + '/FLDR/*.png')
    img_path_list = glob.glob(dir_path + '/LDR/*.png')
    img_path_list_F.sort()
    img_path_list.sort()

    img_list_H = list()
    img_list_F = list()
    img_list = list()

    if is_upexposure_trained:
        img_order = range(len(img_path_list))
    else:
        img_order = range(len(img_path_list) - 1, -1, -1)

    # read images, including HDR images, FLDR images and LDR images
    img_H_ = img_io.readHDR(img_path_list_H[0], FLAGS.height, FLAGS.width)
    img_list_H.append(np.squeeze(img_H_))
    img_list_H = np.array(img_list_H)

    for j in img_order:
        img_F = img_io.readLDR(img_path_list_F[j], FLAGS.height, FLAGS.width)
        img_list_F.append(np.squeeze(img_F))

        img = img_io.readLDR(img_path_list[j], FLAGS.height, FLAGS.width)
        img_list.append(np.squeeze(img))

    img_list_F = np.array(img_list_F)
    img_list = np.array(img_list)

    return img_list_H, img_list_F, img_list

def myPerTrain(FLAGS, input_frame_id, lossmask_list, img_list_H, img_list_F, img_list):
    start_frame_id = input_frame_id + 2
    end_frame_id = min(start_frame_id + FLAGS.predicted_window_len, len(img_list))
    # input LDR or FLDR
    x_batch_l = np.array([img_list[input_frame_id, :, :, :]])
    x_batch_fl = np.array([img_list_F[input_frame_id, :, :, :]])

    y_batch_0 = img_list_H.reshape((1,) + x_batch_l.shape[:]).astype(np.float32)
    y_batch_1 = np.array(
        [img_list[start_frame_id:end_frame_id, :, :, :]])
    y_batch = np.concatenate([y_batch_0, y_batch_1], axis=1)

    dummy_len = FLAGS.predicted_window_len - y_batch.shape[1]
    zero_dummy = np.zeros(x_batch_l.size * dummy_len).reshape(
        y_batch.shape[:1] + (dummy_len,) + y_batch.shape[2:]).astype(
        np.float32)
    y_batch = np.concatenate([y_batch, zero_dummy], axis=1)

    # reshape
    y_batch = np.reshape(y_batch, (FLAGS.batch_size, -1, 3))
    lossmask_s = np.reshape(lossmask_list[dummy_len], (FLAGS.batch_size, -1, 3))
    return x_batch_l, x_batch_fl, y_batch, lossmask_s

def myLoadGraph(FLAGS, FLAGS_m):
    gm = tf.Graph()
    sess_gm = tf.Session(graph=gm)
    with gm.as_default():
        if FLAGS.experiment_type == 0:
            out_path = FLAGS_m + '/' + FLAGS.data_name + '/ldr2hdr/'

        else:
            out_path = FLAGS_m + '/' + FLAGS.data_name + '/fldr2hdr/'
        ckpt_gm = tf.train.get_checkpoint_state(out_path)
        if ckpt_gm and ckpt_gm.model_checkpoint_path:
            saver_gm = tf.train.import_meta_graph(ckpt_gm.model_checkpoint_path + '.meta')
            saver_gm.restore(sess_gm, ckpt_gm.model_checkpoint_path)
            pred_placehoder_gm = tf.get_collection('predict')[0]
            gragh_gm = tf.get_default_graph()
            # tensor_name_list_gm = [tensor.name for tensor in gragh_gm.as_graph_def().node]
            x_gm = gragh_gm.get_tensor_by_name('x:0')
    return sess_gm, pred_placehoder_gm, x_gm


def _process(result):
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result

def camera_func_0(x):
    result = 0.02075 * np.power(x, 3) + 0.5034 * np.power(x, 2) + 0.4727 * x - 0.001136
    return _process(result)

def camera_func_1(x):
    result = 0.9491 * np.power(x, 3) - 2.97 * np.power(x, 2) + 3.114 * x - 0.1031
    return _process(result)

def camera_func_2(x):
    result = 0.2108 * np.power(x, 3) - 0.9448 * np.power(x, 2) + 1.711 * x + 0.0246
    return _process(result)

def camera_func_3(x):
    result = 2.909 * np.power(x, 3) - 5.858 * np.power(x, 2) + 3.908 * x + 0.0883
    return _process(result)

def camera_func_4(x):
    result = 1.462 * np.power(x, 3) - 3.16 * np.power(x, 2) + 2.618 * x + 0.1047
    return _process(result)


def exposure_times(tao, T):
    delt_t = list()
    tt = int(T / 2 + 1)
    for t in range(tt):
        delt_t.append(math.pow(1 / tao, t))
    delt_t.reverse()
    for t in range(tt - 1):
        delt_t.append(math.pow(tao, t + 1))
    delt_t = np.array(delt_t)
    return delt_t


def hdr_filter_func(hdr):
    # R G B -> 0.7316,0.6839,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.4945
    temp[:,:,1] = hdr[:,:,1] * 0.6839
    temp[:,:,2] = hdr[:,:,2] * 0.7316
    return temp

def hdr_filter_func1(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.4945
    temp[:,:,1] = hdr[:,:,1] * 0.7316
    temp[:,:,2] = hdr[:,:,2] * 0.6839
    return temp

def hdr_filter_func2(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.7316
    temp[:,:,1] = hdr[:,:,1] * 0.4945
    temp[:,:,2] = hdr[:,:,2] * 0.6839
    return temp

def hdr_filter_func3(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.7316
    temp[:,:,1] = hdr[:,:,1] * 0.6839
    temp[:,:,2] = hdr[:,:,2] * 0.4945
    return temp

def hdr_filter_func4(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.6839
    temp[:,:,1] = hdr[:,:,1] * 0.7316
    temp[:,:,2] = hdr[:,:,2] * 0.4945
    return temp

def hdr_filter_func5(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.6839
    temp[:,:,1] = hdr[:,:,1] * 0.4945
    temp[:,:,2] = hdr[:,:,2] * 0.7316
    return temp

def hdr_filter_func6(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.7316
    temp[:,:,1] = hdr[:,:,1] * 0.7316
    temp[:,:,2] = hdr[:,:,2] * 0.7316
    return temp

def hdr_filter_func7(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.6839
    temp[:,:,1] = hdr[:,:,1] * 0.6839
    temp[:,:,2] = hdr[:,:,2] * 0.6839
    return temp

def hdr_filter_func8(hdr):
    # R G B -> 0.6839,0.7316,0.4945
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.4945
    temp[:,:,1] = hdr[:,:,1] * 0.4945
    temp[:,:,2] = hdr[:,:,2] * 0.4945
    return temp