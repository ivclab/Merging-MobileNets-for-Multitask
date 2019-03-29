from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags

flags.DEFINE_string('out_dir_A', 
                    '../merged_ckpt/model_A/merged_model_A.ckpt', 
                    'specify A model output dir')
flags.DEFINE_string('out_dir_B', 
                    '../merged_ckpt/model_B/merged_model_B.ckpt', 
                    'specify B model output dir')

flags.DEFINE_string('./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/cubs_cropped', 
                    './merged_params_A.npy', 
                    'specify A numpy file input dir')
flags.DEFINE_string('./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/flowers_102', 
                    './merged_params_B.npy', 
                    'specify B numpy file input dir')

flags.DEFINE_string(
    'model_A', 'cubs_cropped', '')

flags.DEFINE_string(
    'model_B', 'flowers_102', '')

flags.DEFINE_string(
    'model_meta_A', './checkpoints/cubs_cropped/model.ckpt-0.meta', '')

flags.DEFINE_string(
    'model_ckpt_A', './checkpoints/cubs_cropped', '')

flags.DEFINE_string(
    'model_meta_B', './checkpoints/flowers_102/model.ckpt-0.meta', '')

flags.DEFINE_string(
    'model_ckpt_B', './checkpoints/flowers_102', '')

flags.DEFINE_string(
    'merged_npy_A', './hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/cubs_cropped/1/merged_cubs_cropped.npy', '')

flags.DEFINE_string(
    'merged_npy_B', './hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/flowers_102/1/merged_flowers_102.npy', '')

flags.DEFINE_string(
    'merged_ckpt_out_dir_A', './hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/cubs_cropped/1/', '')

flags.DEFINE_string(
    'merged_ckpt_out_dir_B', './hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/flowers_102/1/', '')


FLAGS = flags.FLAGS

is_cubs_cropped_first = 1

if is_cubs_cropped_first == 1:
    model_meta_A = "./checkpoints/cubs_cropped/model.ckpt-30000.meta"
    model_ckpt_A = "./checkpoints/cubs_cropped"
    model_meta_B = "./checkpoints/flowers_102/model.ckpt-30000.meta"
    model_ckpt_B = "./checkpoints/flowers_102"

np_value_A = np.load(FLAGS.merged_npy_A)
np_value_B = np.load(FLAGS.merged_npy_B)

merged_value = tf.placeholder(tf.float32)

def check_tensor_equal(tensor_A, tensor_B, size):
    for layer in range(12,13):
        pw_A = tensor_A.item().get('pw_weights')[layer]
        pw_B = tensor_B.item().get('pw_weights')[layer]
        for indx_A in range(size):
            for indx_B in range(size):
                if np.array_equal(pw_A[:,:,:,indx_A],pw_B[:,:,:,indx_B]):
                    print('TURE',indx_A,indx_B)


def assign_value(model_meta, model_ckpt, np_value, model_name, out_dir):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta, clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
        all_vars = tf.global_variables()
        merged_value = tf.placeholder(tf.float32)
        layer_dw_BN_gamma = 0
        layer_dw_BN_beta = 0
        layer_dw_BN_moving_mean = 0
        layer_dw_BN_moving_variance = 0
        layer_dw_weights = 0
        layer_pw_BN_gamma = 0
        layer_pw_BN_beta= 0
        layer_pw_BN_moving_mean = 0
        layer_pw_BN_moving_variance= 0
        layer_pw_weights = 0
        for indx, var in enumerate(all_vars):
            if "depthwise/BatchNorm/gamma" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('dw_BN_gamma')[layer_dw_BN_gamma]})
                layer_dw_BN_gamma+=1
            elif "depthwise/BatchNorm/beta" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('dw_BN_beta')[layer_dw_BN_beta]})
                layer_dw_BN_beta+=1      
            elif "depthwise/BatchNorm/moving_mean" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('dw_BN_moving_mean')[layer_dw_BN_moving_mean]})
                layer_dw_BN_moving_mean+=1
            elif "depthwise/BatchNorm/moving_variance" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('dw_BN_moving_variance')[layer_dw_BN_moving_variance]})
                layer_dw_BN_moving_variance+=1    
            elif "depthwise/depthwise_weights" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('dw_weights')[layer_dw_weights]})
                layer_dw_weights+=1

            elif "pointwise/BatchNorm/gamma" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('pw_BN_gamma')[layer_pw_BN_gamma]})
                layer_pw_BN_gamma+=1
            elif "pointwise/BatchNorm/beta" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('pw_BN_beta')[layer_pw_BN_beta]})
                layer_pw_BN_beta+=1
            elif "pointwise/BatchNorm/moving_mean" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('pw_BN_moving_mean')[layer_pw_BN_moving_mean]})
                layer_pw_BN_moving_mean+=1
            elif "pointwise/BatchNorm/moving_variance" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('pw_BN_moving_variance')[layer_pw_BN_moving_variance]})
                layer_pw_BN_moving_variance+=1
            elif "pointwise/weights" in var.name:
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('pw_weights')[layer_pw_weights]})
                layer_pw_weights+=1
        
        merged_ckpt_dir = out_dir  + 'merged_' + model_name + '.ckpt'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # np.save(merged_ckpt_dir, merged_params_A)
        save_path = saver.save(sess, merged_ckpt_dir)



tf.reset_default_graph()
# check_tensor_equal(np_value_A, np_value_B, 1024)
assign_value(FLAGS.model_meta_A, FLAGS.model_ckpt_A, np_value_A, FLAGS.model_A, FLAGS.merged_ckpt_out_dir_A)
tf.reset_default_graph()
# check_tensor_equal(np_value_A, np_value_B, 1024) 
assign_value(FLAGS.model_meta_B, FLAGS.model_ckpt_B, np_value_B, FLAGS.model_B, FLAGS.merged_ckpt_out_dir_B)



