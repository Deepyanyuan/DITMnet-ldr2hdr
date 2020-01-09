from __future__ import division

# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/13 15:00
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description:

'''
# ----- Load libraries -------------------------------------
import numpy as np
import os, glob, time
import tensorflow as tf
from src import archs as network
from src.ops import myLossMask, myCost2, myPerFile, myPerTrain

# ----- Custom functions -----
def train(FLAGS):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    # define placeholder
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width, 3], name='x')
    y_gt = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, 3], name='y_gt')
    mask = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, 3], name='mask')

    # load model
    model = network.mitm_net(x, is_train=True, batch_size=FLAGS.batch_size, pad='SAME')
    y_out = model.outputs * mask
    norm_y_out = tf.nn.l2_normalize(y_out, axis=2)
    norm_y_gt = tf.nn.l2_normalize(y_gt, axis=2)

    # load cost function
    cost1, cost2, cost = myCost2(FLAGS, y_out, y_gt, norm_y_out, norm_y_gt)

    # define optimizer
    # train_param = model.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(cost)
    # save model output
    tf.add_to_collection('predict', model.outputs)

    # add summary
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('cost1', cost1)
    tf.summary.scalar('cost2', cost2)
    tf.summary.scalar('lr', FLAGS.lr)

    summary_op = tf.summary.merge_all()

    # open session and initialize variables
    saver = tf.train.Saver(max_to_keep=2)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    # init
    sess.run(init)

    # add breakpoint training function
    if FLAGS.learn_type == 0:
        out_path = FLAGS.dm + '/' + FLAGS.data_name
    else:
        out_path = FLAGS.um + '/' + FLAGS.data_name
    if FLAGS.experiment_type == 0:
        out_path = out_path + '/ldr2hdr'
    else:
        out_path = out_path + '/fldr2hdr'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path_summary = out_path + '/' + FLAGS.summary_dir
    if not os.path.exists(out_path_summary):
        os.makedirs(out_path_summary)

    summary_writer = tf.summary.FileWriter(out_path_summary, graph=sess.graph)
    ckpt = tf.train.get_checkpoint_state(out_path + '/')
    if ckpt and ckpt.model_checkpoint_path:
        print('Load pre-model Done')
        # str_step = str(FLAGS.step)

        # if FLAGS.learn_type == 0:
        #     saver.restore(sess, ckpt.model_checkpoint_path + '/down_model' + '-' + str_step)
        # else:
        #     saver.restore(sess, ckpt.model_checkpoint_path + '/up_model' + '-' + str_step)
        saver.restore(sess, ckpt.model_checkpoint_path)

    # begin training
    # images mask
    lossmask_list = myLossMask(FLAGS)

    # images path
    dir_path_list = glob.glob(FLAGS.train_dir + '/' + FLAGS.data_name + '/*')
    dir_path_list = dir_path_list[:]
    N = len(dir_path_list)
    for epoch in range(FLAGS.epoch):
        print('epoch', epoch)
        start = time.clock()
        loss_gen_sum = 0.
        loss_gen_sum1 = 0.
        loss_gen_sum2 = 0.
        perm = np.random.permutation(N)
        for i in range(N):
            # if i % 100 == 0:
            #     print('i', i)
            img_list_H, img_list_F, img_list = myPerFile(FLAGS, dir_path_list, perm, i)
            M = len(img_list)
            for input_frame_id in range(len(img_list) - 1):

                x_batch_l, x_batch_fl, y_batch, lossmask = myPerTrain(FLAGS, input_frame_id, lossmask_list, img_list_H,
                                                                    img_list_F, img_list)

                if FLAGS.experiment_type == 0:
                    my_feed = {x: x_batch_l, y_gt: y_batch, mask: lossmask}        # LDR -> HDR
                else:
                    my_feed = {x: x_batch_fl, y_gt: y_batch, mask: lossmask}    # FLDR -> HDR

                _, loss_gen, loss_gen1, loss_gen2, summary = sess.run(
                    [train_op, cost, cost1, cost2, summary_op],
                    feed_dict=my_feed)

                # save model and params
                global_step = input_frame_id + i * (len(img_list) - 1) + epoch * (len(img_list) - 1) * N
                if (global_step + 1) % FLAGS.step == 0:
                    print('global_step', global_step)
                    summary_writer.add_summary(summary, global_step)
                    summary_writer.flush()

                if (global_step + 1) % (FLAGS.step * 1) == 0:
                    # print('global_step', global_step)
                    if FLAGS.learn_type == 0:
                        saver.save(sess, out_path + '/down_model', global_step=global_step)
                    else:
                        saver.save(sess, out_path + '/up_model', global_step=global_step)
                    print('saved model')

                loss_gen_sum += loss_gen / (M - 1)
                loss_gen_sum1 += loss_gen1 / (M - 1)
                loss_gen_sum2 += loss_gen2 / (M - 1)

        print('loss:', loss_gen_sum / N)
        print('loss1:', loss_gen_sum1 / N)
        print('loss2:', loss_gen_sum2 / N)

        end = time.clock()
        print('each train time is ', end - start)

    sess.close()

    return