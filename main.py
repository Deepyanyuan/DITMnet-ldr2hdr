from __future__ import division

# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2019/11/13 14:58
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description:

'''
# ----- Load libraries -------------------------------------
import tensorflow as tf
from src.train import train
from src.test import test
from src.predict import predict

# ----- Main function -----
if __name__ == '__main__':
    # Beginning running
    FLAGS = tf.flags.FLAGS

    # about model
    tf.flags.DEFINE_integer("model_type", "0", "Model type. (0:train, 1:test, 2:predict")
    tf.flags.DEFINE_integer("step", "2500", "training state per step save the model params, it defaults to 2500")

    # about train
    tf.flags.DEFINE_integer("experiment_type", "1", "Learning type. (0:ldr2hdr, 1:fldr2hdr")
    tf.flags.DEFINE_integer("learn_type", "1", "Learning type. (0:downexposure, 1:upexposure")
    tf.flags.DEFINE_integer("epoch", "100", "epoch defaults to 1000")
    tf.flags.DEFINE_integer("cost1_weight", "24", "cost function 1 defaults to 24")
    tf.flags.DEFINE_integer("cost2_weight", "1", "cost function 2 defaults to 1")
    tf.flags.DEFINE_integer("batch_size", "1", "batch_size defaults to 1")
    tf.flags.DEFINE_float("lr", "0.0001", "lr starts at 0.005, then changes to 0.0001 height")
    tf.flags.DEFINE_integer("predicted_window_len", "8", "predicted_window_len defaults to 8")

    # about predict
    tf.flags.DEFINE_float("alpha", "0.6", "predicted state alpha defaults to 0.6")

    # about image
    tf.flags.DEFINE_integer("width", "256", "Reconstruction image width")
    tf.flags.DEFINE_integer("height", "256", "Reconstruction image height")
    # tf.flags.DEFINE_float("scaling", "1.0",
    #                       "Pre-scaling, which is followed by clipping, in order to remove compression artifacts close to highlights")
    tf.flags.DEFINE_float("gamma", "1.0",
                          "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")

    # about path
    tf.flags.DEFINE_string("data_name", "Fairchild_HDR", "data_name is Fairchild_HDR or Funt_HDR")
    tf.flags.DEFINE_string("train_dir", "./dataset/training_samples",
                           "Path to training image directory or an individual image")
    tf.flags.DEFINE_string("valid_dir", "./dataset/valid_samples",
                           "Path to valid image directory or an individual image")
    tf.flags.DEFINE_string("test_dir", "./dataset/testing_samples",
                           "Path to testing image directory or an individual image")
    tf.flags.DEFINE_string("out_dir", "results", "Path to output directory")
    tf.flags.DEFINE_string("summary_dir", "summary", "Path to output directory")
    tf.flags.DEFINE_string("dm", "./models_dm", "Path to trained CNN dm_weights")
    tf.flags.DEFINE_string("um", "./models_um", "Path to trained CNN um_weights")

    if FLAGS.model_type == 0:
        train(FLAGS)
    elif FLAGS.model_type == 1:
        test(FLAGS)
    else:
        predict(FLAGS)

    # Program over
    print('success! u are a smart boy!')
