from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset_factory
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 236


record_iterator = tf.python_io.tf_record_iterator(
    path='/media/iis/external/dataset/deepfashion/train-01023-of-01024')

for string_record in record_iterator:

  example = tf.train.Example()

  example.ParseFromString(string_record)

  height = int(example.features.feature['image/height']
                               .int64_list
                               .value[0])

  width = int(example.features.feature['image/width']
                              .int64_list
                              .value[0])

  encoded_image = (example.features.feature['image/encoded']
                                .bytes_list
                                .value[0])

  label = (example.features.feature['image/class/label']
                              .int64_list
                              .value[0])




# Initialize all global and local variables
  init_op = tf.group(tf.global_variables_initializer(), 
                   tf.local_variables_initializer())
  

  with tf.Session() as sess:
    feature = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=-1),
      'image/class/width': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=224),
      'image/class/height': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=224),
    }     
     # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(['/media/iis/external/dataset/imagenet2012/train-01000-of-01024'], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
    features={
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=-1),
      'image/class/width': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=224),
      'image/class/height': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=224),
      })
    

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/class/width'], tf.int32)
    width = tf.cast(features['image/class/height'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    
    # Any preprocessing here ...

    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 
                                    dtype=tf.int32)
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH)

    sess.run(init_op)
    for i in range(3):
     img, lab = sess.run([image, label])
     # 檢查每個 batch 的圖片維度
     print(img.shape)
     # 顯示每個 batch 的第一張圖
     io.imshow(img[0, :, :, :])
     plt.show()
