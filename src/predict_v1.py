from __future__ import division

# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/11/13 15:09
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description:

'''
# ----- Load libraries -------------------------------------
import numpy as np
import sys, os, glob, time
import tensorflow as tf
from src.merge_HDR import merge_HDR
from src import img_io
from src.ops import myLoadGraph

# ----- Custom functions -----
def myPreprocessing(FLAGS):
    dir_path_list = glob.glob(FLAGS.test_dir + '/' + FLAGS.data_name + '/*')
    dir_path_list = dir_path_list[:]
    dir_outpath = glob.glob(FLAGS.out_dir + '/' + FLAGS.data_name)
    if not os.path.exists(FLAGS.out_dir + '/' + FLAGS.data_name):
        os.makedirs(FLAGS.out_dir + '/' + FLAGS.data_name)
    return dir_path_list, dir_outpath


def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()


def predict(FLAGS):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    # GPU set
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu90%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)

    print("\nStarting to load multiple graphs...\n\n")

    dm = tf.Graph()
    um = tf.Graph()

    sess_dm = tf.Session(graph=dm)
    sess_um = tf.Session(graph=um)

    with dm.as_default():
        if FLAGS.experiment_type == 0:
            out_path_dm = FLAGS.dm + '/' + FLAGS.data_name + '/ldr2hdr/'

        else:
            out_path_dm = FLAGS.dm + '/' + FLAGS.data_name + '/fldr2hdr/'
        ckpt_dm = tf.train.get_checkpoint_state(out_path_dm)
        if ckpt_dm and ckpt_dm.model_checkpoint_path:
            saver_dm = tf.train.import_meta_graph(ckpt_dm.model_checkpoint_path + '.meta')  # 加载网络图
            saver_dm.restore(sess_dm, ckpt_dm.model_checkpoint_path)  # 加载参数
            pred_placehoder_dm = tf.get_collection('predict')[0]
            gragh_dm = tf.get_default_graph()
            # tensor_name_list_dm = [tensor.name for tensor in gragh_dm.as_graph_def().node]# 得到当前图中所有变量的名称
            # print('tensor_name_list_dm',tensor_name_list_dm)
            x_dm = gragh_dm.get_tensor_by_name('x:0')

    with um.as_default():
        if FLAGS.experiment_type == 0:
            out_path_um = FLAGS.um + '/' + FLAGS.data_name + '/ldr2hdr/'

        else:
            out_path_um = FLAGS.um + '/' + FLAGS.data_name + '/fldr2hdr/'
        ckpt_um = tf.train.get_checkpoint_state(out_path_um)
        if ckpt_um and ckpt_um.model_checkpoint_path:
            saver_um = tf.train.import_meta_graph(ckpt_um.model_checkpoint_path + '.meta')
            saver_um.restore(sess_um, ckpt_um.model_checkpoint_path)
            pred_placehoder_um = tf.get_collection('predict')[0]
            gragh_um = tf.get_default_graph()
            x_um = gragh_um.get_tensor_by_name('x:0')


    # load images to predict
    # load training images
    dir_path_list, dir_outpath = myPreprocessing(FLAGS)
    print('dir_outpath', dir_outpath)
    N = len(dir_path_list)
    for i in range(N):
        start_time = time.clock()
        dir_path = dir_path_list[i]

        if FLAGS.experiment_type == 0:
            frams = [glob.glob(dir_path + '/LDR/1.png')[0], glob.glob(dir_path + '/LDR/4.png')[0],
                     glob.glob(dir_path + '/LDR/7.png')[0]]
        else:
            frams = [glob.glob(dir_path + '/FLDR/1.png')[0], glob.glob(dir_path + '/FLDR/4.png')[0],
                     glob.glob(dir_path + '/FLDR/7.png')[0]]

        filename_root = os.path.basename(dir_path)
        print('filename', filename_root)
        save_path = dir_outpath[0] + '/' + filename_root
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        try:
            x_input_1 = img_io.readLDR(frams[0], FLAGS.height, FLAGS.width)
            x_input_2 = img_io.readLDR(frams[1], FLAGS.height, FLAGS.width)
            x_input_3 = img_io.readLDR(frams[2], FLAGS.height, FLAGS.width)
            # print('x_input_1.shape', x_input_1.shape)

            y_dm_1 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_1})
            y_dm_2 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_2})
            y_dm_3 = sess_dm.run(pred_placehoder_dm, feed_dict={x_dm: x_input_3})
            # print('y_dm_1.shape', y_dm_1.shape)

            y_um_1 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_1})
            y_um_2 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_2})
            y_um_3 = sess_um.run(pred_placehoder_um, feed_dict={x_um: x_input_3})
            # print('y_um_1.shape', y_um_1.shape)

            y_1 = merge_HDR(y_dm_1, y_um_1, x_input_1, alpha=FLAGS.alpha)
            y_2 = merge_HDR(y_dm_2, y_um_2, x_input_2, alpha=FLAGS.alpha)
            y_3 = merge_HDR(y_dm_3, y_um_3, x_input_3, alpha=FLAGS.alpha)

            # y_1 = np.power(np.maximum(y_1, 0.0), FLAGS.gamma)
            # y_2 = np.power(np.maximum(y_2, 0.0), FLAGS.gamma)
            # y_3 = np.power(np.maximum(y_3, 0.0), FLAGS.gamma)


            img_io.writeEXR(y_1, '%s/HDR_DITMnet_1.exr' % save_path)
            img_io.writeEXR(y_2, '%s/HDR_DITMnet_4.exr' % save_path)
            img_io.writeEXR(y_3, '%s/HDR_DITMnet_7.exr' % save_path)

            print('\tdone\n')
            end_time = time.clock()
            print('a single image needs: ', end_time - start_time)

        except img_io.IOException as e:
            print_("\n\t\tWarning! ", 'w', True)
            print_("%s\n" % e, 'w')
        except Exception as e:
            print_("\n\t\tError: ", 'e', True)
            print_("%s\n" % e, 'e')

    sess.close()
    print('Done!\n')
    return