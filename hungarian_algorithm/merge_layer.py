from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.optimize import linear_sum_assignment


flags = tf.app.flags

flags.DEFINE_integer('start_layer', 0, 'specify the initial layer to merge')
flags.DEFINE_integer('end_layer', 13, 'specify the final layer to merge')

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
    'merged_npy_out_dir_A', './hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/cubs_cropped/', '')

flags.DEFINE_string(
    'merged_npy_out_dir_B', './hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/flowers_102/', '')

FLAGS = flags.FLAGS

dw_weights_A = []
dw_BN_gamma_A = []
dw_BN_beta_A = []
pw_weights_A = []
pw_BN_gamma_A = []
pw_BN_beta_A = []
fc_weights_A = []

dw_weights_B = []
dw_BN_gamma_B = []
dw_BN_beta_B = []
pw_weights_B = []
pw_BN_gamma_B = []
pw_BN_beta_B = []
fc_weights_B = []

is_cubs_cropped_first = 1
is_reverse = 0

if is_cubs_cropped_first == 1:
    model_meta_A = "./checkpoints/cubs_cropped/model.ckpt-30000.meta"
    model_ckpt_A = "./checkpoints/cubs_cropped"
    model_meta_B = "./checkpoints/flowers_102/model.ckpt-30000.meta"
    model_ckpt_B = "./checkpoints/flowers_102"

np_params_A = []
np_params_B = []


def check_tensor_equal(tensor_A, tensor_B, size):
    for indx_A in range(size):
        for indx_B in range(size):
            if np.array_equal(tensor_A[:,:,:,indx_A],tensor_B[:,:,:,indx_B]):
                print('TURE',indx_A,indx_B)


def convert_ckpt_to_npy(model_meta, model_ckpt):
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
    conv_2d_weights = []
    conv_2d_BN_gamma = []
    conv_2d_BN_beta = []
    conv_2d_BN_moving_mean = []
    conv_2d_BN_moving_variance = []   

    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta, clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
        # all_vars = tf.trainable_variables(scope=model_scope)
        all_vars = tf.global_variables()

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

            elif "depthwise/BatchNorm/gamma" in var.name:
                dw_BN_gamma.append(var.eval())
            elif "depthwise/BatchNorm/beta" in var.name:
                dw_BN_beta.append(var.eval())
            elif "depthwise/BatchNorm/moving_mean" in var.name:
                dw_BN_moving_mean.append(var.eval())
            elif "depthwise/BatchNorm/moving_variance" in var.name:
                dw_BN_moving_variance.append(var.eval())
            elif "depthwise/depthwise_weights" in var.name:
                dw_weights.append(var.eval())

            elif "depthwise_1/BatchNorm/gamma" in var.name:
                dw_BN_gamma.append(var.eval())
            elif "depthwise_1/BatchNorm/beta" in var.name:
                dw_BN_beta.append(var.eval())
            elif "depthwise_1/BatchNorm/moving_mean" in var.name:
                dw_BN_moving_mean.append(var.eval())
            elif "depthwise_1/BatchNorm/moving_variance" in var.name:
                dw_BN_moving_variance.append(var.eval())
            elif "depthwise_1/depthwise_weights" in var.name:
                dw_weights.append(var.eval())

            elif "pointwise/BatchNorm/gamma" in var.name:
                pw_BN_gamma.append(var.eval())
            elif "pointwise/BatchNorm/beta" in var.name:
                pw_BN_beta.append(var.eval())
            elif "pointwise/BatchNorm/moving_mean" in var.name:
                pw_BN_moving_mean.append(var.eval())
            elif "pointwise/BatchNorm/moving_variance" in var.name:
                pw_BN_moving_variance.append(var.eval())
            elif "pointwise/weights" in var.name:
                pw_weights.append(var.eval())

            elif "Logits/Conv2d_1c_1x1/weights" in var.name:
                fc_weights.append(var.eval())
            elif "Logits/Conv2d_1_1c_1x1/weights" in var.name:
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                fc_weights.append(var.eval())     

        
        dict_npy_params = dict()
        dict_npy_params['conv_2d_weights'] = conv_2d_weights
        dict_npy_params['conv_2d_BN_beta'] = conv_2d_BN_beta
        dict_npy_params['conv_2d_BN_gamma'] = conv_2d_BN_gamma

        dict_npy_params['dw_BN_gamma'] = dw_BN_gamma
        dict_npy_params['dw_BN_beta'] = dw_BN_beta
        dict_npy_params['dw_BN_moving_mean'] = dw_BN_moving_mean
        dict_npy_params['dw_BN_moving_variance'] = dw_BN_moving_variance

        dict_npy_params['dw_weights'] = dw_weights
        dict_npy_params['pw_BN_gamma'] = pw_BN_gamma
        dict_npy_params['pw_BN_beta'] = pw_BN_beta
        dict_npy_params['pw_BN_moving_mean'] = pw_BN_moving_mean
        dict_npy_params['pw_BN_moving_variance'] = pw_BN_moving_variance
        dict_npy_params['pw_weights'] = pw_weights

        dict_npy_params['fc_weights'] = fc_weights
    
        return dict_npy_params

def build_cost_matrix(weights_A, weights_B, layer):
    out_channel_size = (weights_A['pw_weights'][layer][0,0,0,:]).size
    cost_matrix = np.empty((0, out_channel_size), dtype="float32")
    for iter_A in range(out_channel_size):
        dist = []
        dist_array = []
        tmp = []
        layer_weights_A = weights_A['pw_weights'][layer][:,:,:,iter_A]
        for iter_B in range(out_channel_size):
            layer_weights_B = weights_B['pw_weights'][layer][:,:,:,iter_B]        
            dist = np.linalg.norm(np.absolute(layer_weights_A - layer_weights_B))
            dist_array = np.append(dist_array, dist)
        cost_matrix = np.vstack((cost_matrix, dist_array))
    return cost_matrix

def hungarian_matching(weights_A, weights_B, cost_matrix, layer):
    changed_merged_channels = []
    merged_channels = []
    original_indx = []
    changed_indx = []
    merged_BN_gamma = []
    merged_BN_beta = []
    merged_BN_moving_mean = []
    merged_BN_moving_variance = []
    changed_merged_BN_gamma = []
    changed_merged_BN_beta = []
    changed_merged_BN_moving_mean = []
    changed_merged_BN_moving_variance = []  
    num_BN_channel = weights_A['pw_BN_gamma'][layer][:].size
    num_pointwise_channel = weights_A['pw_weights'][layer][0,0,:,0].size
    changed_merged_channels = np.empty((1,
                                        1,
                                        num_pointwise_channel,num_BN_channel), 
                                        dtype="float32")
    merged_channels = np.empty((1,1,num_pointwise_channel,0), dtype="float32") 
    merged_BN_gamma = np.empty((0), dtype="float32")    
    merged_BN_beta = np.empty((0), dtype="float32")
    merged_BN_moving_mean = np.empty((0), dtype="float32")    
    merged_BN_moving_variance = np.empty((0), dtype="float32")
    changed_merged_BN_gamma = np.empty((num_BN_channel), dtype="float32") 
    changed_merged_BN_beta = np.empty((num_BN_channel), dtype="float32")
    changed_merged_BN_moving_mean = np.empty((num_BN_channel), dtype="float32") 
    changed_merged_BN_moving_variance = np.empty((num_BN_channel), dtype="float32")
    original_indx, changed_indx = linear_sum_assignment(cost_matrix)
    # no shuffle
    changed_indx = original_indx
    # changed_indx = np.roll(changed_indx, 1)
    if is_cubs_cropped_first == 0:
        tmp = original_indx
        original_indx = changed_indx
        changed_indx = tmp
    
    num_channel = len(original_indx)
    for num in range(num_channel):
        one_channel_A = weights_A['pw_weights'][layer][:,:,:,original_indx[num]]
        one_channel_B = weights_B['pw_weights'][layer][:,:,:,changed_indx[num]]
        one_merged_channel = np.mean([one_channel_A,one_channel_B],
                                        axis=0)
        one_merged_channel = np.expand_dims(one_merged_channel, axis=3)
        merged_channels = np.concatenate((merged_channels,one_merged_channel), 
                                            axis=3) 

    # Mapping merged channels back to the original index
    if is_cubs_cropped_first == 0:
        tmp = original_indx
        original_indx = changed_indx
        changed_indx = tmp 
    # no shuffle
    changed_indx = original_indx

    for num in range(num_channel):
        changed_merged_channels\
        [:,:,:,changed_indx[num]] = merged_channels[:,:,:,original_indx[num]]

    for num in range(num_channel):
        BN_gamma_A = weights_A['pw_BN_gamma'][layer][original_indx[num]]
        BN_gamma_B = weights_B['pw_BN_gamma'][layer][changed_indx[num]]
        BN_moving_mean_A = weights_A['pw_BN_moving_mean'][layer][original_indx[num]]
        BN_moving_mean_B = weights_B['pw_BN_moving_mean'][layer][changed_indx[num]]
        BN_beta_A = weights_A['pw_BN_beta'][layer][original_indx[num]]
        BN_beta_B = weights_B['pw_BN_beta'][layer][changed_indx[num]]
        BN_moving_variance_A = weights_A['pw_BN_moving_variance'][layer][original_indx[num]]
        BN_moving_variance_B = weights_B['pw_BN_moving_variance'][layer][changed_indx[num]]

        one_merged_BN_gamma = np.mean([BN_gamma_A,BN_gamma_B],
                                        axis=0)                       
        one_merged_BN_beta = np.mean([BN_beta_A,BN_beta_B],
                                        axis=0)
        one_merged_BN_moving_mean = np.mean([BN_moving_mean_A,BN_moving_mean_B],
                                        axis=0)                       
        one_merged_BN_moving_variance = np.mean([BN_moving_variance_A,BN_moving_variance_B],
                                        axis=0)
        one_merged_BN_gamma = np.expand_dims(one_merged_BN_gamma, axis=0)
        one_merged_BN_beta = np.expand_dims(one_merged_BN_beta, axis=0)
        one_merged_BN_moving_mean = np.expand_dims(one_merged_BN_moving_mean, axis=0)
        one_merged_BN_moving_variance = np.expand_dims(one_merged_BN_moving_variance, axis=0)        
        merged_BN_gamma = np.concatenate((merged_BN_gamma,one_merged_BN_gamma), 
                                        axis=0) 
        merged_BN_beta = np.concatenate((merged_BN_beta,one_merged_BN_beta), 
                                        axis=0)
        merged_BN_moving_mean = np.concatenate((merged_BN_moving_mean,one_merged_BN_moving_mean), 
                                        axis=0) 
        merged_BN_moving_variance = np.concatenate((merged_BN_moving_variance,one_merged_BN_moving_variance), 
                                        axis=0) 
    # Mapping merged channels back to the original index 
    for num in range(num_BN_channel):
        changed_merged_BN_gamma[changed_indx[num]] = merged_BN_gamma[original_indx[num]]
        changed_merged_BN_beta[changed_indx[num]] = merged_BN_beta[original_indx[num]]
        changed_merged_BN_moving_mean[changed_indx[num]] = merged_BN_moving_mean[original_indx[num]]
        changed_merged_BN_moving_variance[changed_indx[num]] = merged_BN_moving_variance[original_indx[num]] 
    
    
    return (changed_merged_channels, merged_channels, 
            original_indx, changed_indx,
            changed_merged_BN_gamma, changed_merged_BN_beta,
            changed_merged_BN_moving_mean, changed_merged_BN_moving_variance,
            merged_BN_gamma, merged_BN_beta,
            merged_BN_moving_mean, merged_BN_moving_variance
            )   

def rearrange_next_depthwise_channel(weights_B, pw_original_indx, pw_changed_indx, layer):
    dw_next_BN_gamma = []
    dw_next_BN_beta = []
    dw_next_BN_moving_mean = []
    dw_next_BN_moving_variance = []
    dw_next_weight = []
    rearranged_dw_next_BN_gamma = []
    rearranged_dw_next_BN_beta = []
    rearranged_dw_next_BN_moving_mean = []
    rearranged_dw_next_BN_moving_variance = []
    rearranged_dw_next_weight = []
    dw_next_BN_gamma = weights_B['dw_BN_gamma'][layer+1]
    dw_next_BN_beta = weights_B['dw_BN_beta'][layer+1]
    dw_next_BN_moving_mean= weights_B['dw_BN_moving_mean'][layer+1]
    dw_next_BN_moving_variance = weights_B['dw_BN_moving_variance'][layer+1]
    dw_next_weight = weights_B['dw_weights'][layer+1]
    rearranged_dw_next_BN_gamma = weights_B['dw_BN_gamma'][layer+1]
    rearranged_dw_next_BN_beta = weights_B['dw_BN_beta'][layer+1]
    rearranged_dw_next_BN_moving_mean = weights_B['dw_BN_moving_mean'][layer+1]
    rearranged_dw_next_BN_moving_variance= weights_B['dw_BN_moving_variance'][layer+1]
    rearranged_dw_next_weight = weights_B['dw_weights'][layer+1]
    num_channel = pw_original_indx.size
    # for i in pw_original_indx:
    #     print('pw_original_indx',i)
    # for i in pw_changed_indx:
    #     print('pw_changed_indx',i)
    
    if layer == 12:
        return 'ERROR!!! OUT OF SIZE'
    for num in range(num_channel):
        rearranged_dw_next_weight[:,:,pw_changed_indx[num],:] = \
        dw_next_weight[:,:,pw_original_indx[num],:]
    for num in range(num_channel):
        rearranged_dw_next_BN_gamma[pw_changed_indx[num]] = \
        dw_next_BN_gamma[pw_original_indx[num]]
    for num in range(num_channel):
        rearranged_dw_next_BN_beta[pw_changed_indx[num]] = \
        dw_next_BN_beta[pw_original_indx[num]]
    for num in range(num_channel):
        rearranged_dw_next_BN_moving_mean[pw_changed_indx[num]] = \
        dw_next_BN_moving_mean[pw_original_indx[num]]
    for num in range(num_channel):
        rearranged_dw_next_BN_moving_variance[pw_changed_indx[num]] = \
        dw_next_BN_moving_variance[pw_original_indx[num]]
    
    return rearranged_dw_next_weight,rearranged_dw_next_BN_gamma, rearranged_dw_next_BN_beta, rearranged_dw_next_BN_moving_mean, rearranged_dw_next_BN_moving_variance

def merge_fc_layer(weights_A, weights_B, fc_pw_original_indx, fc_pw_changed_indx):
    fc_weights_A = []
    rearranged_fc_weights_A = []
    num_layer = len(fc_pw_original_indx)
    fc_weights_A = weights_A['fc_weights'][0]
    rearranged_fc_weights_A = weights_A['fc_weights'][0]
    num_channel = len(fc_pw_original_indx)
    for num in range(num_channel):
        rearranged_fc_weights_A[:,:,fc_pw_changed_indx[num],:] = \
        fc_weights_A[:,:,fc_pw_original_indx[num],:] 
    
    fc_weights_B = []
    rearranged_fc_weights_B = []
    num_layer = len(fc_pw_original_indx)
    fc_weights_B = weights_B['fc_weights'][0]
    rearranged_fc_weights_B = weights_B['fc_weights'][0]
    num_channel = len(fc_pw_original_indx)
    for num in range(num_channel):
        rearranged_fc_weights_B[:,:,fc_pw_changed_indx[num],:] = \
        fc_weights_B[:,:,fc_pw_original_indx[num],:]

    
    return rearranged_fc_weights_A, rearranged_fc_weights_B 

model_scope_A = None
model_scope_B = None

# model_scope_A = 'MobilenetV1_M'
# model_scope_B = 'MobilenetV1_M'
tf.reset_default_graph()
np_params_A = convert_ckpt_to_npy(FLAGS.model_meta_A, FLAGS.model_ckpt_A)
tf.reset_default_graph()
np_params_B = convert_ckpt_to_npy(FLAGS.model_meta_B, FLAGS.model_ckpt_B)                                          

print('A',np_params_A['fc_weights'][0].shape)
print('B',np_params_B['fc_weights'][0].shape)

def run_merging_params(start_layer, end_layer):
    rearranged_dp_next_layer = []
    dw_layer_cost_matrix = []
    rearranged_dp_layer = []
    rearrange_np_params_B = []
    rearranged_fc_weights_A = []
    rearranged_fc_weights_B = []
    rearranged_pw_previous_weight = []
    rearranged_pw_previous_BN_gamma = []
    rearranged_pw_previous_BN_beta = []
    merged_params_A = []
    merged_params_B = []
    merged_params_A = np_params_A
    merged_params_B = np_params_B
    is_start = 1
    is_start_pw_layer = 1
    is_end_pw_layer = 0
    for layer in range(start_layer, end_layer, 1): 

        print(layer)
        if layer == 12:
            is_end_pw_layer = 1


        pw_layer_cost_matrix = build_cost_matrix(np_params_A, 
                                                 np_params_B,
                                                 layer,)

        pw_changed_merged_channels, \
        pw_merged_channels, \
        pw_original_indx, \
        pw_changed_indx, \
        pw_changed_merged_BN_gamma, \
        pw_changed_merged_BN_beta, \
        pw_changed_merged_BN_moving_mean, \
        pw_changed_merged_BN_moving_variance, \
        pw_merged_BN_gamma, \
        pw_merged_BN_beta, \
        pw_merged_BN_moving_mean, \
        pw_merged_BN_moving_variance = hungarian_matching(np_params_A, 
                                                          np_params_B, 
                                                          pw_layer_cost_matrix, 
                                                          layer)

                                                                                                        
        if is_end_pw_layer == 0:
            print("NOT in the end_pw")
            if is_cubs_cropped_first == 1: 
                rearranged_dw_next_weight, \
                rearranged_dw_next_BN_gamma, \
                rearranged_dw_next_BN_beta, \
                rearranged_dw_next_BN_moving_mean, \
                rearranged_dw_next_BN_moving_variance = rearrange_next_depthwise_channel(np_params_B, 
                                                                            pw_original_indx, 
                                                                            pw_changed_indx,
                                                                            layer)
            elif is_cubs_cropped_first == 0:
                rearranged_dw_next_weight, \
                rearranged_dw_next_BN_gamma, \
                rearranged_dw_next_BN_beta, \
                rearranged_dw_next_BN_moving_mean, \
                rearranged_dw_next_BN_moving_variance = rearrange_next_depthwise_channel(np_params_A, 
                                                                            pw_original_indx, 
                                                                            pw_changed_indx,
                                                                            layer)    

            merged_params_B['dw_weights'][layer+1] = rearranged_dw_next_weight
            merged_params_B['dw_BN_gamma'][layer+1] = rearranged_dw_next_BN_gamma
            merged_params_B['dw_BN_beta'][layer+1] = rearranged_dw_next_BN_beta
            merged_params_B['dw_BN_moving_mean'][layer+1] = rearranged_dw_next_BN_moving_mean
            merged_params_B['dw_BN_moving_variance'][layer+1] = rearranged_dw_next_BN_moving_variance
        
        # if is_end_pw_layer == 1:
        #     rearranged_fc_weights_A, rearranged_fc_weights_B = merge_fc_layer(np_params_A, 
        #                                                                       rearrange_np_params_B, 
        #                                                                       pw_original_indx, 
        #                                                                       pw_changed_indx)
        #     print('fc_A',rearranged_fc_weights_A.shape)
        #     print('fc_B',rearranged_fc_weights_B.shape)

                
        if is_cubs_cropped_first == 1:
            merged_params_A['pw_BN_gamma'][layer] = pw_merged_BN_gamma
            merged_params_A['pw_BN_beta'][layer] = pw_merged_BN_beta
            merged_params_A['pw_BN_moving_mean'][layer] = pw_merged_BN_moving_mean
            merged_params_A['pw_BN_moving_variance'][layer] = pw_merged_BN_moving_variance
            merged_params_A['pw_weights'][layer] = pw_merged_channels
            
            merged_params_B['pw_BN_gamma'][layer] = pw_changed_merged_BN_gamma
            merged_params_B['pw_BN_beta'][layer] = pw_changed_merged_BN_beta
            merged_params_B['pw_BN_moving_mean'][layer] = pw_merged_BN_moving_mean
            merged_params_B['pw_BN_moving_variance'][layer] = pw_merged_BN_moving_variance
            merged_params_B['pw_weights'][layer] = pw_changed_merged_channels
        if is_cubs_cropped_first == 0:
            merged_params_A['pw_BN_gamma'][layer] = pw_changed_merged_BN_gamma
            merged_params_A['pw_BN_beta'][layer] = pw_changed_merged_BN_beta
            merged_params_A['pw_BN_moving_mean'][layer] = pw_merged_BN_moving_mean
            merged_params_A['pw_BN_moving_variance'][layer] = pw_merged_BN_moving_variance
            merged_params_A['pw_weights'][layer] = pw_changed_merged_channels
    
            merged_params_B['pw_BN_gamma'][layer] = pw_merged_BN_gamma
            merged_params_B['pw_BN_beta'][layer] = pw_merged_BN_beta
            merged_params_B['pw_BN_moving_mean'][layer] = pw_merged_BN_moving_mean
            merged_params_B['pw_BN_moving_variance'][layer] = pw_merged_BN_moving_variance
            merged_params_B['pw_weights'][layer] = pw_merged_channels
        
   
        print('fc_empty_A')
        print('before merged_params_A[fc_weights]',merged_params_A['fc_weights'][0].shape)
        # merged_params_A['fc_weights'][0] = rearranged_fc_weights_A
        print('after merged_params_A[fc_weights][0]',merged_params_A['fc_weights'][0].shape)      
    
        print('fc_empty_B')
        print('before merged_params_B[fc_weights]',merged_params_B['fc_weights'][0].shape)
        # merged_params_B['fc_weights'][0] = rearranged_fc_weights_B
        print('after merged_params_B[fc_weights][0]',merged_params_B['fc_weights'][0].shape)
        
    merged_npy_dir = FLAGS.merged_npy_out_dir_A + str(end_layer)  + '/' + 'merged_' + FLAGS.model_A + '.npy'
    directory = FLAGS.merged_npy_out_dir_A + str(end_layer)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(merged_npy_dir, merged_params_A)

    merged_npy_dir = FLAGS.merged_npy_out_dir_B + str(end_layer)  + '/' + 'merged_' + FLAGS.model_B + '.npy'
    directory = FLAGS.merged_npy_out_dir_B + str(end_layer)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(merged_npy_dir, merged_params_B)

    # check_tensor_equal(merged_params_A['pw_weights'][layer],  merged_params_B['pw_weights'][layer], 64)

def main(unused_arg):
    run_merging_params(FLAGS.start_layer, FLAGS.end_layer)

if __name__ == '__main__':
  tf.app.run(main)