from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


flags = tf.app.flags


flags.DEFINE_string(
    'model_meta', './checkpoints/flowers_102/model.ckpt-0.meta', '')

flags.DEFINE_string(
    'model_ckpt', './checkpoints/flowers_102', '')

flags.DEFINE_string(
    'npy_our_dir', './hungarian_algorithm/origin_npy/flowers_102/flowers_102.npy', '')

flags.DEFINE_string(
    'merged_ckpt_our_dir_A', './hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/cubs_cropped/', '')

flags.DEFINE_string(
    'merged_ckpt_our_dir_B', './hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/flowers_102/', '')




FLAGS = flags.FLAGS


def convert_ckpt_to_npy(model_meta, model_ckpt):

    conv_2d_weights = []
    conv_2d_BN_gamma = []
    conv_2d_BN_beta = []
    conv_2d_BN_moving_mean = []
    conv_2d_BN_moving_variance = []

    dw_weights = []
    dw_BN_gamma = []
    dw_BN_beta = []
    dw_BN_moving_mean = []
    dw_BN_moving_variance = []    

    pw_weights = []
    pw_BN_gamma = []
    pw_BN_beta = []
    pw_BN_moving_mean = []
    pw_BN_moving_variance = []    

    fc_weights = []


    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta, clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
        all_vars = tf.trainable_variables()
        for var in all_vars:
            print(var)

        for indx, var in enumerate(all_vars):
            if "Conv2d_0/weights" in var.name:
                conv_2d_weights.append(var.eval())
            elif "Conv2d_0/BatchNorm/gamma" in var.name:
                conv_2d_BN_gamma.append(var.eval())
            elif "Conv2d_0/BatchNorm/beta" in var.name:
                conv_2d_BN_beta.append(var.eval())
            elif "Conv2d_0/BatchNorm/moving_mean" in var.name:
                conv_2d_BN_moving_mean.append(var.eval())
            elif "Conv2d_0/BatchNorm/moving_variance" in var.name:
                conv_2d_BN_moving_variance.append(var.eval())

            elif "depthwise/depthwise_weights" in var.name:
                dw_weights.append(var.eval())
            elif "depthwise/BatchNorm/gamma" in var.name:
                dw_BN_gamma.append(var.eval())
            elif "depthwise/BatchNorm/beta" in var.name:
                dw_BN_beta.append(var.eval())
            elif "depthwise/BatchNorm/moving_mean" in var.name:
                dw_BN_moving_mean.append(var.eval())
            elif "depthwise/BatchNorm/moving_variance" in var.name:
                dw_BN_moving_variance.append(var.eval())

            elif "pointwise/weights" in var.name:
                pw_weights.append(var.eval())
            elif "pointwise/BatchNorm/gamma" in var.name:
                pw_BN_gamma.append(var.eval())
            elif "pointwise/BatchNorm/beta" in var.name:
                pw_BN_beta.append(var.eval())
            elif "pointwise/BatchNorm/moving_mean" in var.name:
                pw_BN_moving_mean.append(var.eval())
            elif "pointwise/BatchNorm/moving_variance" in var.name:
                pw_BN_moving_variance.append(var.eval())

            elif "Logits/Conv2d_1c_1x1/weights" in var.name:
                fc_weights.append(var.eval())

        
        dict_npy_params = dict()
        dict_npy_params['conv_2d_weights'] = conv_2d_weights
        dict_npy_params['conv_2d_BN_beta'] = conv_2d_BN_beta
        dict_npy_params['conv_2d_BN_gamma'] = conv_2d_BN_gamma
        dict_npy_params['conv_2d_BN_moving_mean'] = conv_2d_BN_moving_mean
        dict_npy_params['conv_2d_BN_moving_variance'] = conv_2d_BN_moving_variance

        dict_npy_params['dw_weights'] = dw_weights
        dict_npy_params['dw_BN_beta'] = dw_BN_beta
        dict_npy_params['dw_BN_gamma'] = dw_BN_gamma
        dict_npy_params['dw_BN_moving_mean'] = dw_BN_moving_mean
        dict_npy_params['dw_BN_moving_variance'] = dw_BN_moving_variance

        dict_npy_params['pw_weights'] = pw_weights
        dict_npy_params['pw_BN_gamma'] = pw_BN_gamma
        dict_npy_params['pw_BN_beta'] = pw_BN_beta
        dict_npy_params['pw_BN_moving_mean'] = pw_BN_moving_mean
        dict_npy_params['pw_BN_moving_variance'] = pw_BN_moving_variance


        dict_npy_params['fc_weights'] = fc_weights
    
        return dict_npy_params

# # if you want to change the graph
# # please use "tf.reset_default_graph()" first
# # following as a example

np_params = convert_ckpt_to_npy(FLAGS.model_meta, FLAGS.model_ckpt)                                          
tf.reset_default_graph()

np.save(FLAGS.npy_our_dir, np_params)
