# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build and train mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from nets import mobilenet_v1
from preprocessing import preprocessing_factory
from tensorflow.python.platform import tf_logging as logging

#slim = tf.contrib.slim
import tensorflow.contrib.slim as slim
import os
import time

tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="3"

flags = tf.app.flags
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', 2000,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('dataset_dir_A', '', 'Location of dataset')
flags.DEFINE_string('dataset_dir_B', '', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')

flags.DEFINE_string('logits_scope_A', 'Logits', 'Location of dataset')
flags.DEFINE_string('logits_scope_B', 'Logits', 'Location of dataset')
flags.DEFINE_string('conv2d_0_scope_A', 'Conv2d_0', 'Location of dataset')
flags.DEFINE_string('conv2d_0_scope_B', 'Conv2d_0', 'Location of dataset')
flags.DEFINE_string('depthwise_scope_A', '_depthwise', 'Location of dataset')
flags.DEFINE_string('depthwise_scope_B', '_depthwise', 'Location of dataset')
flags.DEFINE_string('pointwise_scope_A', '_pointwise', 'Location of dataset')
flags.DEFINE_string('pointwise_scope_B', '_pointwise', 'Location of dataset')
flags.DEFINE_list('pointwise_merged_mask', [1,1,1,1,1,1,1,1,1,1,1,1,1,], 'How often to save checkpoints, secs')

tf.app.flags.DEFINE_string(
    'checkpoint_path_A', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_path_B', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_path_A_teacher', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_path_B_teacher', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_path_trained_merged', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes_A', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes_B', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes_A', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_string(
    'trainable_scopes_B', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
    'checkpoint_model_scope_A', None,
    'Model scope in the checkpoint. None if the same as the trained model.')

tf.app.flags.DEFINE_string(
    'checkpoint_model_scope_B', None,
    'Model scope in the checkpoint. None if the same as the trained model.')

tf.app.flags.DEFINE_string(
    'model_scope_A', None,
    'Model scope in the checkpoint. None if the same as the trained model.')

tf.app.flags.DEFINE_string(
    'model_scope_B', None,
    'Model scope in the checkpoint. None if the same as the trained model.')

tf.app.flags.DEFINE_string(
    'merged_model_scope', 'MobilenetV1_M',
    'Model scope in the checkpoint. None if the same as the trained model.')


tf.app.flags.DEFINE_string(
    'model_scope_A_teacher', None,
    'Model scope in the checkpoint. None if the same as the trained model.')

tf.app.flags.DEFINE_string(
    'model_scope_B_teacher', None,
    'Model scope in the checkpoint. None if the same as the trained model.')    

tf.app.flags.DEFINE_string(
    'dataset_name_A', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_name_B', 'deepfashion', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name_A', 'validation', 'The name of the train/validation/test split.')

tf.app.flags.DEFINE_string(
    'dataset_split_name_B', 'validation', 'The name of the train/validation/test split.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v1', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')


FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94

num_epochs_per_train = 400
imagenet_size = 209222
batch_size = 64
train_steps = int(imagenet_size / batch_size * num_epochs_per_train)


def get_learning_rate():
  if FLAGS.checkpoint_path_A:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 0.001
  else:
    return 0.045


def get_quant_delay():
  if FLAGS.checkpoint_path:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000


def dataset_input(is_training, dataset_name, dataset_split_name, dataset_dir):
  """Data reader for imagenet.

  Reads in imagenet data and performs pre-processing on the images.

  Args:
     is_training: bool specifying if train or validation dataset is needed.
  Returns:
     A batch of images and labels.
  """
  if is_training:
    dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name,
                                          dataset_dir)
  else:
    dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name,
                                          dataset_dir)

  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2 * FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])
  
  preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
         preprocessing_name,
         is_training=is_training)

  image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

  images, labels = tf.train.batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=4,
      capacity=5 * FLAGS.batch_size)
  labels = slim.one_hot_encoding(labels, dataset.num_classes)
  return images, labels

def mask_checkpoint_exclude_scopes(conv2d_0_scope, depthwise_scope, pointwise_scope, logits_scope):
    exclude_scopes = ''
    for layer in range(0,14):
        if layer == 0:    
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + conv2d_0_scope + ','
        else:
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + depthwise_scope + ','
            if FLAGS.pointwise_merged_mask[layer] == '0':
                exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + pointwise_scope + ','
    
    exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + logits_scope
    return exclude_scopes


def zipper_mask_checkpoint_exclude_scopes(conv2d_0_scope, depthwise_scope, pointwise_scope, logits_scope):
    exclude_scopes = ''
    for layer in range(0,14):
        if layer == 0:    
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + conv2d_0_scope + ','
        else:
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + depthwise_scope + ','
            if FLAGS.pointwise_merged_mask[layer] == '0':
                exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + pointwise_scope + ','
    
    exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + logits_scope
    return exclude_scopes

def zipper_mask_merged_trained_checkpoint_exclude_scopes(conv2d_0_scope, depthwise_scope, pointwise_scope, logits_scope):
    exclude_scopes = ''
    for layer in range(0,14):
        if layer == 0:    
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + conv2d_0_scope + ','
        else:
            exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + depthwise_scope + ','
            if FLAGS.pointwise_merged_mask[layer] == '0':
                exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + pointwise_scope + ','
            if layer < 13:
                if FLAGS.pointwise_merged_mask[layer] == '1' and FLAGS.pointwise_merged_mask[layer+1] == '0':
                    exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + '_pointwise' + ','
            elif layer == 13:
                exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + '_pointwise' + ','

    
    exclude_scopes = exclude_scopes + FLAGS.merged_model_scope + '/' + logits_scope
    return exclude_scopes

def mask_variables_to_train(conv2d_0_scope, depthwise_scope, pointwise_scope, logits_scope):
    variables_scopes = ''
    for layer in range(0,14):
        if layer == 0:    
            variables_scopes = variables_scopes + FLAGS.merged_model_scope + '/' + conv2d_0_scope + ','
        else:
            variables_scopes = variables_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + depthwise_scope + ','
            if FLAGS.pointwise_merged_mask[layer] == '0':
                 variables_scopes = variables_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + pointwise_scope + ','
            else:
                 variables_scopes = variables_scopes + FLAGS.merged_model_scope + '/' + 'Conv2d_' + str(layer) + '_pointwise' + ','

    
    variables_scopes = variables_scopes + FLAGS.merged_model_scope + '/' + logits_scope
    return variables_scopes
 


def _get_variables_to_train(trainable_scopes):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train  


def distill_layers(student_activation, teacher_activation):
    layers_l1_loss = 0
    for end_point in sorted(student_activation):
        if 'pointwise' in end_point:
            if FLAGS.pointwise_scope_A in end_point:
                end_point_teacher = end_point
                end_point_teacher = end_point_teacher.replace(FLAGS.pointwise_scope_A, '_pointwise')
                pointwise_l1_loss = tf.reduce_mean(tf.abs(teacher_activation[end_point_teacher] - student_activation[end_point]))
                layers_l1_loss = layers_l1_loss + pointwise_l1_loss
            elif FLAGS.pointwise_scope_B in end_point:
                end_point_teacher = end_point
                end_point_teacher = end_point_teacher.replace(FLAGS.pointwise_scope_B, '_pointwise')
                pointwise_l1_loss = tf.reduce_mean(tf.abs(teacher_activation[end_point_teacher] - student_activation[end_point]))
                layers_l1_loss = layers_l1_loss + pointwise_l1_loss
            else:
                pointwise_l1_loss = tf.reduce_mean(tf.abs(teacher_activation[end_point] - student_activation[end_point]))
                layers_l1_loss = layers_l1_loss + pointwise_l1_loss
    return layers_l1_loss


def build_model():

  """Builds graph for model to train with rewrites for quantization.

  Returns:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
  """

  mask_variables_to_train_A = mask_variables_to_train(FLAGS.conv2d_0_scope_A,
                                                           FLAGS.depthwise_scope_A,
                                                           FLAGS.pointwise_scope_A, 
                                                           FLAGS.logits_scope_A,)

  mask_variables_to_train_B = mask_variables_to_train(FLAGS.conv2d_0_scope_B, 
                                                           FLAGS.depthwise_scope_B,
                                                           FLAGS.pointwise_scope_B, 
                                                           FLAGS.logits_scope_B,)
  print('#######mask_variables_to_train_A#########')
  print(mask_variables_to_train_A)
  print('#######mask_variables_to_train_B#########')
  print(mask_variables_to_train_B)

  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    inputs_A, labels_A = dataset_input(is_training=True, dataset_name=FLAGS.dataset_name_A, 
                                        dataset_split_name=FLAGS.dataset_split_name_A,
                                        dataset_dir=FLAGS.dataset_dir_A)
    inputs_B, labels_B = dataset_input(is_training=True, dataset_name=FLAGS.dataset_name_B, 
                                        dataset_split_name=FLAGS.dataset_split_name_B,
                                        dataset_dir=FLAGS.dataset_dir_B)
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
      dataset_A = dataset_factory.get_dataset(FLAGS.dataset_name_A, FLAGS.dataset_split_name_A,
                                            FLAGS.dataset_dir_A)
      dataset_B = dataset_factory.get_dataset(FLAGS.dataset_name_B, FLAGS.dataset_split_name_B,
                                            FLAGS.dataset_dir_B)

      logits_A, end_points_A = mobilenet_v1.mobilenet_v1(
          inputs_A,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=dataset_A.num_classes,
          model_scope=FLAGS.model_scope_A_teacher,
          pointwise_merged_mask=['1','1','1','1','1','1','1','1','1','1','1','1','1','1'])
      tf.losses.softmax_cross_entropy(labels_A, logits_A)

      logits_B, end_points_B = mobilenet_v1.mobilenet_v1(
          inputs_B,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=dataset_B.num_classes,
          model_scope=FLAGS.model_scope_B_teacher,
          pointwise_merged_mask=['1','1','1','1','1','1','1','1','1','1','1','1','1','1'],
          )
      tf.losses.softmax_cross_entropy(labels_B, logits_B)
      
      logits_M_A, end_points_M_A = mobilenet_v1.mobilenet_v1(
          inputs_A,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=dataset_A.num_classes,
          model_scope=FLAGS.merged_model_scope,
          logits_scope=FLAGS.logits_scope_A,
          conv2d_0_scope=FLAGS.conv2d_0_scope_A,
          depthwise_scope=FLAGS.depthwise_scope_A,
          pointwise_scope=FLAGS.pointwise_scope_A,
          pointwise_merged_mask=FLAGS.pointwise_merged_mask)
      logits_loss_A = tf.losses.softmax_cross_entropy(labels_A, logits_M_A)

      logits_M_B, end_points_M_B = mobilenet_v1.mobilenet_v1(
          inputs_B,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=dataset_B.num_classes,
          model_scope=FLAGS.merged_model_scope,
          logits_scope=FLAGS.logits_scope_B,
          conv2d_0_scope=FLAGS.conv2d_0_scope_B,
          depthwise_scope=FLAGS.depthwise_scope_B,
          pointwise_scope=FLAGS.pointwise_scope_B,
          pointwise_merged_mask=FLAGS.pointwise_merged_mask,
          reuse=tf.AUTO_REUSE)
      logits_loss_B = tf.losses.softmax_cross_entropy(labels_B, logits_M_B)


      distilling_loss_A = distill_layers(end_points_M_A, end_points_A)
      distilling_loss_B = distill_layers(end_points_M_B, end_points_B)

      total_loss_A = logits_loss_A + distilling_loss_A
      total_loss_B = logits_loss_B + distilling_loss_B

    # Call rewriter to produce graph with fake quant ops and folded batch norms
    # quant_delay delays start of quantization till quant_delay steps, allowing
    # for better model accuracy.
    if FLAGS.quantize:
      tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

    # total_loss = tf.losses.get_total_loss(name='total_loss')

    
    # Configure the learning rate using an exponential decay.

    num_epochs_per_decay_A = 10
    dataset_A = dataset_factory.get_dataset(
        FLAGS.dataset_name_A, FLAGS.dataset_split_name_A, FLAGS.dataset_dir_A)       
    dataset_size_A = dataset_A.num_samples
    decay_steps_A = int(dataset_size_A / FLAGS.batch_size * num_epochs_per_decay_A)

    learning_rate_A = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps_A,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt_A = tf.train.GradientDescentOptimizer(learning_rate_A)

    variables_to_train_A = _get_variables_to_train(mask_variables_to_train_A)

    train_tensor_A = slim.learning.create_train_op(
      total_loss_A,
      optimizer=opt_A,
      variables_to_train=variables_to_train_A) 


    num_epochs_per_decay_B = 10
    dataset_B = dataset_factory.get_dataset(
        FLAGS.dataset_name_B, FLAGS.dataset_split_name_B, FLAGS.dataset_dir_B)       
    dataset_size_B = dataset_B.num_samples
    decay_steps_B = int(dataset_size_B / FLAGS.batch_size * num_epochs_per_decay_B)

    learning_rate_B = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps_B,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt_B = tf.train.GradientDescentOptimizer(learning_rate_B)

    variables_to_train_B = _get_variables_to_train(mask_variables_to_train_B)

    train_tensor_B = slim.learning.create_train_op(
      total_loss_B,
      optimizer=opt_B,
      variables_to_train=variables_to_train_B) 

    train_tensor = [train_tensor_A, train_tensor_B]

    accuracy_A = tf.metrics.accuracy(labels=tf.argmax(labels_A, 1),
                                   predictions=tf.argmax(logits_A,1))
    correct_prediction_A = tf.equal(tf.argmax(labels_A, 1), tf.argmax(logits_A, 1))
    accuracy_batch_A = tf.reduce_mean(tf.cast(correct_prediction_A, tf.float32))

    accuracy_B = tf.metrics.accuracy(labels=tf.argmax(labels_B, 1),
                                   predictions=tf.argmax(logits_B,1))
    correct_prediction_B = tf.equal(tf.argmax(labels_B, 1), tf.argmax(logits_B, 1))
    accuracy_batch_B = tf.reduce_mean(tf.cast(correct_prediction_B, tf.float32))
   
    total_loss = total_loss_A + total_loss_B

    # Variables to train.
    for var in variables_to_train_A:
        print('@A',var)
    for var in variables_to_train_B:
        print('@B',var)
                               
  slim.summaries.add_scalar_summary(logits_loss_A, 'logits_loss_A', 'losses')
  slim.summaries.add_scalar_summary(logits_loss_B, 'logits_loss_B', 'losses')
  slim.summaries.add_scalar_summary(distilling_loss_A, 'distilling_loss_A', 'losses')
  slim.summaries.add_scalar_summary(distilling_loss_B, 'distilling_loss_B', 'losses')
  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate_A, 'learning_rate_A', 'training')
  slim.summaries.add_scalar_summary(learning_rate_A, 'learning_rate_A', 'training')
  slim.summaries.add_scalar_summary(accuracy_A, 'accuracy_A', 'accuracy_A')
  slim.summaries.add_scalar_summary(accuracy_batch_A, 'accuracy_batch_A', 'accuracy_A')
  slim.summaries.add_scalar_summary(accuracy_B, 'accuracy_B', 'accuracy_B')
  slim.summaries.add_scalar_summary(accuracy_batch_B, 'accuracy_batch_B', 'accuracy_B')
  return g, train_tensor


def multiple_train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  total_loss_A, np_global_step = sess.run([train_op[0], global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  total_loss_B, np_global_step = sess.run([train_op[1], global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss_A = %.4f, loss_B = %.4f (%.3f sec/step)',
                   np_global_step, total_loss_A, total_loss_B, time_elapsed)

  total_loss = total_loss_A + total_loss_B

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


def _get_init_fn(train_dir, 
                 checkpoint_exclude_scopes, 
                 model_scope, 
                 checkpoint_model_scope, 
                 checkpoint_path, 
                 logits_scope, 
                 Conv2d_0_endpoint, 
                 depthwise_scope, 
                 pointwise_scope,
                 pointwise_merged_mask):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """

  if not os.path.exists(FLAGS.train_dir):
    print('***Create new dir***',train_dir)
    os.makedirs(FLAGS.train_dir)

  if checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % train_dir)
    return None

  exclusions = []
  if checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables(model_scope):
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
    else:
      variables_to_restore.append(var)
  # Change model scope if necessary.
  def replace_variables_name(var):
    var_name = var.op.name
    var_name = var_name.replace(model_scope, checkpoint_model_scope)
    if logits_scope != 'Logits':
        var_name = var_name.replace(logits_scope, 'Logits')
    if Conv2d_0_endpoint != 'Conv2d_0':
        var_name = var_name.replace(Conv2d_0_endpoint, 'Conv2d_0')
    if depthwise_scope != '_depthwise':
        var_name = var_name.replace(depthwise_scope, '_depthwise')
    if pointwise_scope != '_pointwise':
        var_name = var_name.replace(pointwise_scope, '_pointwise')
    return var_name
   
  variables_to_restore = {
    replace_variables_name(var):var
    for var in variables_to_restore}
  

  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
  else:
    checkpoint_path = checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)

def multiple_init_fn():

  exclude_scopes_A = mask_checkpoint_exclude_scopes(FLAGS.conv2d_0_scope_B,
                                                  FLAGS.depthwise_scope_B,
                                                  FLAGS.pointwise_scope_B, 
                                                  FLAGS.logits_scope_B,)

  exclude_scopes_B = mask_checkpoint_exclude_scopes(FLAGS.conv2d_0_scope_A, 
                                                    FLAGS.depthwise_scope_A,
                                                    FLAGS.pointwise_scope_A, 
                                                    FLAGS.logits_scope_A)


  exclude_trained_merged_scope_A = zipper_mask_merged_trained_checkpoint_exclude_scopes(FLAGS.conv2d_0_scope_B,
                                                  FLAGS.depthwise_scope_B,
                                                  FLAGS.pointwise_scope_B, 
                                                  FLAGS.logits_scope_B,)

  exclude_trained_merged_scope_B = zipper_mask_merged_trained_checkpoint_exclude_scopes(FLAGS.conv2d_0_scope_A, 
                                                    FLAGS.depthwise_scope_A,
                                                    FLAGS.pointwise_scope_A, 
                                                    FLAGS.logits_scope_A)


  
  All_exclude_trained_merged_scope = exclude_trained_merged_scope_A + ',' + exclude_trained_merged_scope_B
  print('All_exclude_trained_merged_scope',All_exclude_trained_merged_scope)
  

  slim_init_fn_A = _get_init_fn(
                              train_dir=FLAGS.train_dir, 
                              checkpoint_exclude_scopes=None, 
                              model_scope=FLAGS.model_scope_A_teacher,  
                              checkpoint_model_scope=FLAGS.checkpoint_model_scope_A, 
                              checkpoint_path=FLAGS.checkpoint_path_A_teacher,
                              Conv2d_0_endpoint='Conv2d_0',
                              depthwise_scope='_depthwise',
                              pointwise_scope='_pointwise',
                              logits_scope='Logits',
                              pointwise_merged_mask=FLAGS.pointwise_merged_mask )

  slim_init_fn_B = _get_init_fn(
                              train_dir=FLAGS.train_dir, 
                              checkpoint_exclude_scopes=None, 
                              model_scope=FLAGS.model_scope_B_teacher,  
                              checkpoint_model_scope=FLAGS.checkpoint_model_scope_B, 
                              checkpoint_path=FLAGS.checkpoint_path_B_teacher,
                              Conv2d_0_endpoint='Conv2d_0',
                              depthwise_scope='_depthwise',
                              pointwise_scope='_pointwise',
                              logits_scope='Logits',
                              pointwise_merged_mask=FLAGS.pointwise_merged_mask )


  slim_init_fn_M_A = _get_init_fn(
                              train_dir=FLAGS.train_dir, 
                              checkpoint_exclude_scopes=exclude_scopes_A, 
                              model_scope=FLAGS.merged_model_scope,  
                              checkpoint_model_scope=FLAGS.checkpoint_model_scope_A, 
                              checkpoint_path=FLAGS.checkpoint_path_A,
                              Conv2d_0_endpoint=FLAGS.conv2d_0_scope_A,
                              depthwise_scope=FLAGS.depthwise_scope_A,
                              pointwise_scope=FLAGS.pointwise_scope_A,
                              logits_scope=FLAGS.logits_scope_A,
                              pointwise_merged_mask=FLAGS.pointwise_merged_mask )

  slim_init_fn_M_B = _get_init_fn(
                              train_dir=FLAGS.train_dir, 
                              checkpoint_exclude_scopes=exclude_scopes_B, 
                              model_scope=FLAGS.merged_model_scope,  
                              checkpoint_model_scope=FLAGS.checkpoint_model_scope_B, 
                              checkpoint_path=FLAGS.checkpoint_path_B,
                              Conv2d_0_endpoint=FLAGS.conv2d_0_scope_B,
                              depthwise_scope=FLAGS.depthwise_scope_B,
                              pointwise_scope=FLAGS.pointwise_scope_B,
                              logits_scope=FLAGS.logits_scope_B,
                              pointwise_merged_mask=FLAGS.pointwise_merged_mask )

  slim_init_fn_trained_M = _get_init_fn(
                              train_dir=FLAGS.train_dir, 
                              checkpoint_exclude_scopes=All_exclude_trained_merged_scope, 
                              model_scope=FLAGS.merged_model_scope,  
                              checkpoint_model_scope=FLAGS.merged_model_scope, 
                              checkpoint_path=FLAGS.checkpoint_path_trained_merged,
                              Conv2d_0_endpoint='Conv2d_0',
                              depthwise_scope='_depthwise',
                              pointwise_scope='_pointwise',
                              logits_scope='Logits',
                              pointwise_merged_mask=FLAGS.pointwise_merged_mask )

      
        # If we are restoring from a floating point model, we need to initialize
        # the global step to zero for the exponential decay to result in
        # reasonable learning rates.

  def init_fn(sess):
    slim_init_fn_A(sess)
    slim_init_fn_B(sess)
    slim_init_fn_M_A(sess)
    slim_init_fn_M_B(sess)
    if FLAGS.checkpoint_path_trained_merged: 
        slim_init_fn_trained_M(sess)
    elif FLAGS.pointwise_merged_mask[1] == '1' and FLAGS.pointwise_merged_mask[2]  == '0':
        print('Merged Layer 1 without Trained-merged Checkpoint')
    else:
        print('!!! Not Zipper or No Trained-merged Checkpoint !!!')


  
  return init_fn 


def train_model():
  """Trains mobilenet_v1."""
  g, train_tensor = build_model()
  with g.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(log_device_placement=False,
                              gpu_options=gpu_options)     
    slim.learning.train(
        train_tensor,
        FLAGS.train_dir,
        train_step_fn=multiple_train_step,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        # number_of_steps=train_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=multiple_init_fn(),
        session_config=config,
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)
