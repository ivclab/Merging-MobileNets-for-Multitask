from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.optimize import linear_sum_assignment

from tensorflow.python import debug as tf_debug

flags = tf.app.flags

flags.DEFINE_integer('start_layer', 0, 'specify the initial layer to merge')
flags.DEFINE_integer('end_layer', 15, 'specify the final layer to merge')
flags.DEFINE_string('out_dir_A', 
                    './merged_ckpt/imagenet/merged_imagenet.ckpt', 
                    'specify A model output dir')
flags.DEFINE_string('out_dir_B',
                    './merged_ckpt/cubs_cropped/merged_cubs.ckpt', 
                    'specify B model output dir')

flags.DEFINE_string('np_dir_A', 
                    './imagenet_npy/imagenet.npy', 
                    'specify A numpy file input dir')
flags.DEFINE_string('np_dir_B', 
                    './cubs_cropped_npy/cubs_cropped.npy', 
                    'specify B numpy file input dir')

flags.DEFINE_string('model_meta_A', 
                    '../logs/resnet_v2_50/imagenet/model.ckpt-0.meta/', 
                    'specify A numpy file input dir')
flags.DEFINE_string('model_meta_B', 
                    '../logs/resnet_v2_50/cubs_cropped/all/model.ckpt-5000.meta', 
                    'specify B numpy file input dir')


flags.DEFINE_string('model_ckpt_A', 
                    '../logs/resnet_v2_50/imagenet/', 
                    'specify A numpy file input dir')
flags.DEFINE_string('model_ckpt_B', 
                    '../logs/resnet_v2_50/cubs_cropped/all/', 
                    'specify B numpy file input dir')                   

FLAGS = flags.FLAGS





def merge_conv_block(conv_block_A, conv_block_B):
    merged_conv_block = []
    num_conv = len(conv_block_A)
    for layer in range(num_conv):    
        one_merged_conv_2d_kernels = []
        merged_conv_2d_kernels = []
        one_merged_conv_2d_biases = []
        merged_conv_2d_biases = []
        if conv_block_A[layer].size > 2048:
            print(conv_block_A[layer].shape)
            h_kernel = len(conv_block_A[layer][:,0,0,0])
            w_kernel = len(conv_block_A[layer][0,:,0,0])
            num_in_channels = len(conv_block_A[layer][0,0,:,0])
            num_out_channels = len(conv_block_A[layer][0,0,0,:])
            merged_conv_2d_kernels = np.empty((h_kernel,w_kernel,num_in_channels,0), dtype="float32") 
            for channel in range(num_out_channels):
                one_conv_2d_kernels_A = conv_block_A[layer][:,:,:,channel]
                one_conv_2d_kernels_B = conv_block_B[layer][:,:,:,channel]
                one_merged_conv_2d_kernels = np.mean([one_conv_2d_kernels_A,one_conv_2d_kernels_B],
                                                        axis=0)
                one_merged_conv_2d_kernels = np.expand_dims(one_merged_conv_2d_kernels, axis=3)
                merged_conv_2d_kernels = np.concatenate((merged_conv_2d_kernels,one_merged_conv_2d_kernels), 
                                                        axis=3)

            merged_conv_block.append(merged_conv_2d_kernels)    
        else:
            print(conv_block_A[layer].shape)
            num_channels = len(conv_block_A[layer][:])
            merged_conv_2d_biases = np.empty((0), dtype="float32") 
            for channel in range(num_channels):
                one_conv_2d_biases_A = conv_block_A[layer][channel]
                one_conv_2d_biases_B = conv_block_B[layer][channel]
                one_merged_conv_2d_biases = np.mean([one_conv_2d_biases_A,one_conv_2d_biases_B],
                                                    axis=0)
                one_merged_conv_2d_biases = np.expand_dims(one_merged_conv_2d_biases, axis=0)
                merged_conv_2d_biases = np.concatenate((merged_conv_2d_biases,one_merged_conv_2d_biases),
                                                    axis=0)

            merged_conv_block.append(merged_conv_2d_biases)


    
    return merged_conv_block 



def merge_npy(np_weights_A, np_weights_B, mask):
    merged_block1_unit_1 = []
    merged_block1_unit_2 = []
    merged_block1_unit_3 = []
    merged_block2_unit_1 = []
    merged_block2_unit_2 = []
    merged_block2_unit_3 = []
    merged_block2_unit_4 = []
    merged_block3_unit_1 = []
    merged_block3_unit_2 = []
    merged_block3_unit_3 = []
    merged_block3_unit_4 = []
    merged_block3_unit_5 = []
    merged_block3_unit_6 = []
    merged_block4_unit_1 = []
    merged_block4_unit_2 = []
    merged_block4_unit_3 = []

    if mask[0] == 1:
        conv1_A = np_weights_A.item().get('block1/unit_1')
        conv1_B = np_weights_B.item().get('block1/unit_1')
        merged_block1_unit_1 = merge_conv_block(conv1_A, conv1_B)
    if mask[1] == 1:
        conv2_A = np_weights_A.item().get('block1/unit_2')
        conv2_B = np_weights_B.item().get('block1/unit_2')
        merged_block1_unit_2 = merge_conv_block(conv2_A, conv2_B)
    if mask[2] == 1:
        conv3_A = np_weights_A.item().get('block1/unit_3')
        conv3_B = np_weights_B.item().get('block1/unit_3')
        merged_block1_unit_3 = merge_conv_block(conv3_A, conv3_B)
    if mask[3] == 1:
        conv4_A = np_weights_A.item().get('block2/unit_1')
        conv4_B = np_weights_B.item().get('block2/unit_1')
        merged_block2_unit_1 = merge_conv_block(conv4_A, conv4_B)
    if mask[4] == 1:
        conv5_A = np_weights_A.item().get('block2/unit_2')
        conv5_B = np_weights_B.item().get('block2/unit_2')
        merged_block2_unit_2 = merge_conv_block(conv5_A, conv5_B)
    if mask[5] == 1:
        conv5_A = np_weights_A.item().get('block2/unit_3')
        conv5_B = np_weights_B.item().get('block2/unit_3')
        merged_block2_unit_3 = merge_conv_block(conv5_A, conv5_B)
    if mask[6] == 1:
        conv5_A = np_weights_A.item().get('block2/unit_4')
        conv5_B = np_weights_B.item().get('block2/unit_4')
        merged_block2_unit_4 = merge_conv_block(conv5_A, conv5_B)
    if mask[7] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_1')
        conv5_B = np_weights_B.item().get('block3/unit_1')
        merged_block3_unit_1 = merge_conv_block(conv5_A, conv5_B)
    if mask[8] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_2')
        conv5_B = np_weights_B.item().get('block3/unit_2')
        merged_block3_unit_2 = merge_conv_block(conv5_A, conv5_B)
    if mask[9] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_3')
        conv5_B = np_weights_B.item().get('block3/unit_3')
        merged_block3_unit_3 = merge_conv_block(conv5_A, conv5_B)
    if mask[10] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_4')
        conv5_B = np_weights_B.item().get('block3/unit_4')
        merged_block3_unit_4 = merge_conv_block(conv5_A, conv5_B)
    if mask[11] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_5')
        conv5_B = np_weights_B.item().get('block3/unit_5')
        merged_block3_unit_5 = merge_conv_block(conv5_A, conv5_B)
    if mask[12] == 1:
        conv5_A = np_weights_A.item().get('block3/unit_6')
        conv5_B = np_weights_B.item().get('block3/unit_6')
        merged_block3_unit_6 = merge_conv_block(conv5_A, conv5_B)
    if mask[13] == 1:
        conv5_A = np_weights_A.item().get('block4/unit_1')
        conv5_B = np_weights_B.item().get('block4/unit_1')
        merged_block4_unit_1 = merge_conv_block(conv5_A, conv5_B)
    if mask[14] == 1:
        conv5_A = np_weights_A.item().get('block4/unit_2')
        conv5_B = np_weights_B.item().get('block4/unit_2')
        merged_block4_unit_2 = merge_conv_block(conv5_A, conv5_B)
    if mask[15] == 1:
        conv5_A = np_weights_A.item().get('block4/unit_3')
        conv5_B = np_weights_B.item().get('block4/unit_3')
        merged_block4_unit_3 = merge_conv_block(conv5_A, conv5_B)



    dict_npy_params = dict()
    dict_npy_params['block1/unit_1'] = merged_block1_unit_1
    dict_npy_params['block1/unit_2'] = merged_block1_unit_2
    dict_npy_params['block1/unit_3'] = merged_block1_unit_3
    dict_npy_params['block2/unit_1'] = merged_block2_unit_1
    dict_npy_params['block2/unit_2'] = merged_block2_unit_2
    dict_npy_params['block2/unit_3'] = merged_block2_unit_3
    dict_npy_params['block2/unit_4'] = merged_block2_unit_4
    dict_npy_params['block3/unit_1'] = merged_block3_unit_1
    dict_npy_params['block3/unit_2'] = merged_block3_unit_2
    dict_npy_params['block3/unit_3'] = merged_block3_unit_3
    dict_npy_params['block3/unit_4'] = merged_block3_unit_4
    dict_npy_params['block3/unit_5'] = merged_block3_unit_5
    dict_npy_params['block3/unit_6'] = merged_block3_unit_6
    dict_npy_params['block4/unit_1'] = merged_block4_unit_1
    dict_npy_params['block4/unit_2'] = merged_block4_unit_2
    dict_npy_params['block4/unit_3'] = merged_block4_unit_3
        
    return dict_npy_params
    

def assign_value(model_meta, model_ckpt, np_value, out_dir):
    layer_conv_2d_kernels = 0
    layer_conv_2d_biases = 0
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.import_meta_graph(model_meta, clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_ckpt))
        all_vars = tf.global_variables()
        for var in all_vars:
            print(var)
        merged_value = tf.placeholder(tf.float32)
        block1_unit_1_layer = 0
        block1_unit_2_layer = 0
        block1_unit_3_layer = 0
        block2_unit_1_layer = 0
        block2_unit_2_layer = 0
        block2_unit_3_layer = 0
        block2_unit_4_layer = 0
        block3_unit_1_layer = 0
        block3_unit_2_layer = 0
        block3_unit_3_layer = 0
        block3_unit_4_layer = 0
        block3_unit_5_layer = 0
        block3_unit_6_layer = 0
        block4_unit_1_layer = 0
        block4_unit_2_layer = 0
        block4_unit_3_layer = 0
        for indx, var in enumerate(all_vars):
            if 'block1/unit_1' in var.name:
                #print(np_value.item().get('block1/unit_1')[block1_unit_1_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block1/unit_1')[block1_unit_1_layer]})
                block1_unit_1_layer = block1_unit_1_layer + 1
                print('block1/unit_1',block1_unit_1_layer)
                print('block1/unit_1',var)
            elif 'block1/unit_2' in var.name:
                #print(np_value.item().get('block1/unit_2')[block1_unit_2_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block1/unit_2')[block1_unit_2_layer]})
                block1_unit_2_layer = block1_unit_2_layer + 1
                print('block1/unit_2',block1_unit_2_layer)
                print('block1/unit_2',var)
            elif 'block1/unit_3' in var.name:
                #print(np_value.item().get('block1/unit_3')[block1_unit_3_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block1/unit_3')[block1_unit_3_layer]})
                block1_unit_3_layer = block1_unit_3_layer + 1
                print('block1/unit_3',block1_unit_3_layer)
                print('block1/unit_3',var)
            elif 'block2/unit_1' in var.name:
                #print(np_value.item().get('block2/unit_1')[block2_unit_1_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block2/unit_1')[block2_unit_1_layer]})
                block2_unit_1_layer = block2_unit_1_layer + 1
                print('block2/unit_1',block2_unit_1_layer)
                print('block2/unit_1',var)   
            elif 'vblock2/unit_2' in var.name:
                #print(np_value.item().get('block2/unit_2')[block2_unit_2_layer]) 
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block2/unit_2')[block2_unit_2_layer]})
                block2_unit_2_layer = block2_unit_2_layer + 1
                print('block2/unit_2',block2_unit_2_layer)
                print('block2/unit_2',var)           
            elif 'block2/unit_3' in var.name:
                #print(np_value.item().get('block2/unit_3')[block2_unit_3_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block2/unit_3')[block2_unit_3_layer]})
                block2_unit_3_layer = block2_unit_3_layer + 1
                print('block2/unit_3',block2_unit_3_layer)
                print('block2/unit_3',var)
            elif 'block2/unit_4' in var.name:
                #print(np_value.item().get('block2/unit_4')[block2_unit_4_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block2/unit_4')[block2_unit_4_layer]})
                block2_unit_4_layer = block2_unit_4_layer + 1
                print('block2/unit_4',block2_unit_4_layer)
                print('block2/unit_4',var)
            elif 'block3/unit_1' in var.name:
                #print(np_value.item().get('block3/unit_1')[block3_unit_1_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_1')[block3_unit_1_layer]})
                block3_unit_1_layer = block3_unit_1_layer + 1
                print('block3/unit_1',block3_unit_1_layer)
                print('block3/unit_1',var)
            elif 'block3/unit_2' in var.name:
                #print(np_value.item().get('block3/unit_2')[block3_unit_2_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_2')[block3_unit_2_layer]})
                block3_unit_2_layer = block3_unit_2_layer + 1
                print('block3/unit_2',block3_unit_2_layer)
                print('block3/unit_2',var)          
            elif 'block3/unit_3' in var.name:
                #print(np_value.item().get('block3/unit_3')[block3_unit_3_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_3')[block3_unit_3_layer]})
                block3_unit_3_layer = block3_unit_3_layer + 1
                print('block3/unit_3',block3_unit_3_layer)
                print('block3/unit_3',var)
            elif 'block3/unit_4' in var.name:
                #print(np_value.item().get('block3/unit_4')[block3_unit_4_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_4')[block3_unit_4_layer]})
                block3_unit_4_layer = block3_unit_4_layer + 1
                print('block3/unit_4',block3_unit_4_layer)
                print('block3/unit_4',var)
            elif 'block3/unit_5' in var.name:
                #print(np_value.item().get('block3/unit_5')[block3_unit_5_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_5')[block3_unit_5_layer]})
                block3_unit_5_layer = block3_unit_5_layer + 1
                print('block3/unit_5',block3_unit_5_layer)
                print('block3/unit_5',var)
            elif 'block3/unit_6' in var.name:
                #print(np_value.item().get('block3/unit_6')[block3_unit_6_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block3/unit_6')[block3_unit_6_layer]})
                block3_unit_6_layer = block3_unit_6_layer + 1
                print('block3/unit_6',block3_unit_6_layer)
                print('block3/unit_6',var)
            elif 'block4/unit_1' in var.name:
                #print(np_value.item().get('block4/unit_1')[block4_unit_1_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block4/unit_1')[block4_unit_1_layer]})
                block4_unit_1_layer = block4_unit_1_layer + 1
                print('block4/unit_1',block4_unit_1_layer)
                print('block4/unit_1',var)
            elif 'block4/unit_2' in var.name:
                #print(np_value.item().get('block4/unit_2')[block4_unit_2_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block4/unit_2')[block4_unit_2_layer]})
                block4_unit_2_layer = block4_unit_2_layer + 1
                print('block4/unit_2',block4_unit_2_layer)
                print('block4/unit_2',var)
            elif 'block4/unit_3' in var.name:
                #print(np_value.item().get('block4/unit_3')[block4_unit_3_layer])
                ops = tf.assign(var, merged_value)
                sess.run(ops, feed_dict={merged_value: \
                np_value.item().get('block4/unit_3')[block4_unit_3_layer]})
                block4_unit_3_layer = block4_unit_3_layer + 1
                print('block4/unit_3',block4_unit_3_layer)
                print('block4/unit_3',var)


        save_path = saver.save(sess, out_dir)



np_weights_imagenet = np.load('./imagenet_npy/imagenet.npy')
np_weights_cubs = np.load('./cubs_cropped_npy/cubs_cropped.npy')
np_weights_flowers = np.load('./flowers_npy/flowers.npy')

model_meta_A = '../logs/resnet_v2_50/imagenet/model.ckpt-0.meta/'
model_ckpt_A = '../logs/resnet_v2_50/imagenet/'
model_meta_B = '../logs/resnet_v2_50/cubs_cropped/all/model.ckpt-5000.meta'
model_ckpt_B = '../logs/resnet_v2_50/cubs_cropped/all/'

model_meta_C = '../logs/resnet_v2_50/flowers_102/all/model.ckpt-5000.meta'
model_ckpt_C = '../logs/resnet_v2_50/flowers_102/all/'

model_meta_D = '../logs/resnet_v2_50/deepfashion/all/model.ckpt-20000.meta'
model_ckpt_D = '../logs/resnet_v2_50/deepfashion/all/'



def main(unused_arg):
    mask = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    merged_npy = merge_npy(np_weights_imagenet, np_weights_cubs, mask)
    np.save('./merged_npy/imagenet/merged_imagenet.npy', merged_npy)
    np.save('./merged_npy/deepfashion/merged_deepfashion.npy', merged_npy)


    np_value_A = np.load('./merged_npy/imagenet/merged_imagenet.npy')
    np_value_B = np.load('./merged_npy/cubs_cropped/merged_cubs.npy')
    np_value_C = np.load('./merged_npy/flowers/merged_flowers.npy')
    np_value_D = np.load('./merged_npy/deepfashion/merged_deepfashion.npy')

    out_merged_ckpt_A = './merged_ckpt/imagenet/merged_imagenet.ckpt'
    out_merged_ckpt_B = './merged_ckpt/cubs_cropped/merged_cubs.ckpt'
    out_merged_ckpt_C = './merged_ckpt/flowers/merged_flowers.ckpt'
    out_merged_ckpt_D = './merged_ckpt/deepfashion/merged_deepfashion.ckpt'
    tf.reset_default_graph()
    assign_value(model_meta_A, model_ckpt_A, np_value_A, out_merged_ckpt_A)
    tf.reset_default_graph()
    assign_value(model_meta_D, model_ckpt_D, np_value_D, out_merged_ckpt_D)

if __name__ == '__main__':
  tf.app.run(main)