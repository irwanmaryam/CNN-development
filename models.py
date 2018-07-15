from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np


#trauncated_normal untuk declare parameter
#tf.nn.dropout untuk kira dropout
#tf.nn.matmul untuk calculation kat dalam fullconnectedlayer
#tf.nn.relu for retified unit
#tf.nn.max_pool for down-sampling
#kalau rectified unit ada 3, so kene buat down sampling 2
#tf.zeros buat tensor dgn semua element yg ada kepada kosong.
#dropout is for control the overfitting in the neural network


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }
  print(fingerprint_size)

def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
 
 
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
is_training, runtime_settings)
  elif model_architecture == 'firsty_model':
    return firsty_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'secondy_model':
    return secondy_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'thirdy_model':
    return thirdy_model(fingerprint_input, model_settings,is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, "low_latency_svdf" , "firsty_model", and "secondy_model"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.
  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def firsty_model(fingerprint_input, model_settings, is_training):



#    with dropout
#    "SAME": output size is the same than input size. This requires the filter window to slip outside input map, hence the need to pad.
#   "VALID": Filter window stays at valid position inside input map, so output size shrinks by filter_size - 1. No padding occurs.


  if is_training:
    dropoutp = tf.placeholder(tf.float32, name='dropout_prob')
  
   
  frequency = model_settings['dct_coefficient_count']
  time = model_settings['spectrogram_length']
  #convert into image
  fingerprint = tf.reshape(fingerprint_input,
                              [-1, time, frequency, 1])
  print(fingerprint)
  
  #Variable declaration
  stridefilterw = 1
  stridefilterh = 1
  height = time
  width = 30
  count_f = 64
  size_ph = 2
  size_pw = 2
  label = model_settings['label_count']
  fullconnectedlayer_channel = 120
  fullconnectedlayer_channel2 = 100
  
  #weight
  
  weights = tf.Variable(tf.truncated_normal([height, width, 1, count_f], stddev=0.01))

  print(weights)
  #bias
  bias = tf.Variable(tf.zeros([count_f]))
  
  convolution_1 = tf.nn.conv2d(fingerprint, weights, [1, stridefilterh, stridefilterw, 1], 'VALID') + bias
  print(convolution_1)
  rectified = tf.nn.relu(convolution_1)
  
  
  print(rectified) 
  if is_training:
  	dropout = tf.nn.dropout(rectified, dropoutp)
	print(dropout)
  else:
	dropout = rectified
	print(dropout)
  pool1 = tf.nn.max_pool(dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  

  grey = np.mean(fingerprint,-1)
  test = tf.Session()
  test.run(tf.global_variables_initializer())
  imagefilter = test.run(convolution_1, feed_dict={fingerprint: grey.reshape(1, time, frequency, 1)})

  tf.reset_default_graph()
  
  

 #display feature maps in rectified
  with tf.variable_scope('Convolutional_1'):
	convolved_image = convolution_1[0:1, 0:20, 0:20, 0:3]
	x_convolutional = tf.transpose(convolved_image, [3, 1, 2, 0])
	tf.summary.image('convolved', x_convolutional)
  with tf.variable_scope('Rectified_Linear_Unit'):
	image = rectified[0:1, 0:20, 0:20, 0:3]
	x_rectified = tf.transpose(image, [3, 1, 2, 0])
	tf.summary.image('image_output', x_rectified)

  with tf.variable_scope('max_pooling'):
	image_pool = pool1[0:1, 0:20, 0:20, 0:3]
	x_pool = tf.transpose(image_pool, [3, 1, 2, 0])
	tf.summary.image('pool_output', x_pool)


  with tf.variable_scope('kernel'):
	kernel = weights[0:1, 0:20, 0:20, 0:3]
	x_kernel = tf.transpose(kernel, [3, 1, 2, 0])
	tf.summary.image('kernelimage', x_kernel)
  


  #[1, size, size, 1], [1, stride, stride, 1], padding]
  weights2 = tf.Variable(tf.truncated_normal([1, 5, count_f, 64],stddev=0.01))
  bias2 = tf.Variable(tf.zeros([count_f]))
  convolution_2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1],'VALID') + bias2
  
  rectified2 = tf.nn.relu(convolution_2)
  if is_training:
 	dropout2 = tf.nn.dropout(rectified2, dropoutp)
  else:
	dropout2 = rectified2
  #the size of feature maps in frequency
  #out_convolution1_width = math.floor((frequency - width + stridefilterw)/ stridefilterw)
  #the size of feature maps in time
  #out_convolution_height = math.floor((time - height + stridefilterh)/ stridefilterh)

  #number of feature maps
  pool2 = tf.nn.max_pool(dropout2,[1, 4, 4, 1], [1, 2, 2, 1],'SAME')
  numberfeaturemaps = int(pool2.shape[1]) * int(pool2.shape[2]) * int(count_f) 
  print(numberfeaturemaps)
  convolutional_flat = tf.reshape(pool2,
                                    [-1, numberfeaturemaps])
  print(convolutional_flat)
  #first layer full
  fullconnectedlayer_w1 = tf.Variable(tf.truncated_normal([numberfeaturemaps, fullconnectedlayer_channel],stddev = 0.01))
  
  
  fullconnectedlayer_b1 = tf.Variable(tf.zeros([fullconnectedlayer_channel]))
  
  fullconnectedlayer1 = tf.matmul(convolutional_flat, fullconnectedlayer_w1) + fullconnectedlayer_b1
  if is_training:
     fullconnectedlayer1_d1 = tf.nn.dropout(fullconnectedlayer1, dropoutp)
  else:
     fullconnectedlayer1_d1 = fullconnectedlayer1
  #second layer full
  fullconnectedlayer_w2 = tf.Variable(tf.truncated_normal([fullconnectedlayer_channel, fullconnectedlayer_channel2], stddev=0.01))
  fullconnectedlayer_b2 = tf.Variable(tf.zeros([fullconnectedlayer_channel2]))
  
  fullconnectedlayer2 = tf.matmul(fullconnectedlayer1_d1, fullconnectedlayer_w2) + fullconnectedlayer_b2

  if is_training:
      fullconnectedlayer2_d2 = tf.nn.dropout(fullconnectedlayer2, dropoutp)
  else:
      fullconnectedlayer2_d2 = fullconnectedlayer2
  

  #layer in classification
  fullconnectedlayer3_w3 = tf.Variable(tf.truncated_normal([fullconnectedlayer_channel2, label], stddev = 0.01))
  fullconnectedlayer3_b3 = tf.Variable(tf.zeros([label]))
  
  fullconnectedlayer3 = tf.matmul(fullconnectedlayer2_d2, fullconnectedlayer3_w3) + fullconnectedlayer3_b3
  if is_training:
    return fullconnectedlayer3,dropoutp
  else:
    return fullconnectedlayer3

"""

  put this backpropagation algorithm in the train.py file  
  ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input))
  lr = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
  gradient = tf.train.GradientDescentOptimizer(
        lr).minimize(ce)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(ground_truth_input, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  confusion_matrix = tf.confusion_matrix(tf.argmax(ground_truth_input,1), tf.argmax(ground_truth_input, 1))
  """

def secondy_model(fingerprint_input, model_settings, is_training):
  

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  frequency = model_settings['dct_coefficient_count']
  time = model_settings['spectrogram_length']
  fingerprint = tf.reshape(fingerprint_input,
                              [-1, time, frequency, 1])
  print(fingerprint)
  print(fingerprint_input)

  
  #Variable declaration
  stridefilterw = 1
  stridefilterh = 1
  height = time
  width = 5
  count_f = 64
  height_p = 2
  width_p = 2
  size_p = 2
  count_s = 64
  count_t = 64
  label_count = model_settings['label_count']
  
  #weights
  weights = tf.Variable(tf.truncated_normal([height, width, 1, count_f], stddev=0.01))
  #bias
  bias = tf.Variable(tf.zeros([count_f]))
  #convolve 1
  convolution_1 = tf.nn.conv2d(fingerprint, weights, [1, stridefilterx, stridefiltery, 1], 'SAME') + bias
  print(convolution_1)
  #rectified unit 1
  rectified = tf.nn.relu(convolution_1)
  print(rectified)

  
  
  if is_training:
      dropout = tf.nn.dropout(rectified, dropout_prob)
  else:
     dropout = rectified
  #downsampling 1
  pool1 = tf.nn.max_pool(dropout, [1, size_p, size_p, 1], [1, height_p, width_p, 1],'SAME')
  weights2 = tf.Variable(tf.truncated_normal([height_p, width_p, count_f, count_s], stddev=0.01))
  bias2 = tf.Variable(tf.zeros([count_s]))
 
  #convolve 2
  convolution_2 = tf.nn.conv2d(pool1, weights2, [1, width_p, height_p, 1], 'SAME') + bias2
  print(convolution_2)
  #rectified unit 2
  rectified2 = tf.nn.relu(convolution_2)
  if is_training:
      dropout2 = tf.nn.dropout(rectified2, dropout_prob)
  else:
     dropout2 = rectified2
  #downsampling 2
  pool2 = tf.nn.max_pool(dropout, [1, size_p, size_p, 1], [1, height_p, width_p, 1],'SAME')
  weights3 = tf.Variable(tf.truncated_normal([height_p, width_p, count_s, count_t], stddev=0.01))
  bias3 = tf.Variable(tf.zeros([count_t]))
  #convolve 3
  convolution_3 = tf.nn.conv2d(pool2, weights3, [1, width_p, height_p, 1], 'SAME') + bias3
  print(convolution_3)
  #retified unit 3
  rectified3 = tf.nn.relu(convolution_3)
  #try nnt
  shape_model = dropout2.get_shape()
  output_width = shape_model[2]
  output_height = shape_model[1]
  numberfeaturemaps = int(
      output_width * output_height *
count_t)
  print(numberfeaturemaps)
  convolution_flat = tf.reshape(dropout2,
                                     [-1, numberfeaturemaps])
  
  fullconnectedlayer_weights = tf.Variable(
      tf.truncated_normal(
          [numberfeaturemaps, label_count], stddev=0.01))
  fullconnectedlayer_bias = tf.Variable(tf.zeros([label_count]))
  fullconnectedlayer = tf.matmul(convolution_flat, fullconnectedlayer_weights) + fullconnectedlayer_bias
  print(fullconnectedlayer)
  if is_training:
    return fullconnectedlayer, dropout_prob
  else:
    return fullconnectedlayer


def thirdy_model(fingerprint_input, model_settings,is_training):



#    without dropout
#    "SAME": output size is the same than input size. This requires the filter window to slip outside input map, hence the need to pad.
#   "VALID": Filter window stays at valid position inside input map, so output size shrinks by filter_size - 1. No padding occurs.

  if is_training:
   dropout = tf.placeholder(tf.float32, name='dropout_prob')
  
  frequency = model_settings['dct_coefficient_count']
  time = model_settings['spectrogram_length']
  #convert into image
  fingerprint = tf.reshape(fingerprint_input,
                              [-1, time, frequency, 1])
  print(fingerprint)
  
  #Variable declaration
  stridefilterw = 4
  stridefilterh = 4
  height = time
  width = 8
  count_f = 128
  size_ph = 3
  size_pw = 3
  label = model_settings['label_count']
  fullconnectedlayer_channel = 120
  fullconnectedlayer_channel2 = 100
  drop_out = 0.5
  #weight
  weights = tf.Variable(tf.truncated_normal([height, width, 1, count_f], stddev=0.01))

  print(weights)
  #bias
  bias = tf.Variable(tf.zeros([count_f]))
  convolution_1 = tf.nn.conv2d(fingerprint, weights, [1, stridefilterh, stridefilterw, 1], 'VALID') + bias
  print(convolution_1)
  rectified = tf.nn.relu(convolution_1)
  #[1, size, size, 1], [1, stride, stride, 1], padding]
 
  if is_training:
	dropout1 = tf.nn.dropout(rectified,dropout)
  else:
	dropout1 = rectified
  pool1 = tf.nn.max_pool(dropout1, [1, size_ph, size_pw, 1], [1, 3, 3, 1],'SAME')
  weights2 = tf.Variable(tf.truncated_normal([1, 3, count_f, 128],stddev=0.01))
  bias2 = tf.Variable(tf.zeros([count_f]))
  convolution_2 = tf.nn.conv2d(pool1, weights2, [1, 3, 3, 1],'VALID') + bias2
  rectified2 = tf.nn.relu(convolution_2)
  if is_training:
	dropout2 = tf.nn.dropout(rectified2,dropout)
  else:
	dropout2 = rectified2
  
  #the size of feature maps in frequency
  #out_convolution1_width = math.floor((frequency - width + stridefilterw)/ stridefilterw)
  #the size of feature maps in time
  #out_convolution_height = math.floor((time - height + stridefilterh)/ stridefilterh)

  #number of feature maps
  numberfeaturemaps =  int(pool2.shape[1]) * int(pool2.shape[2]) * int(count_f)
  print(numberfeaturemaps)
  #first layer full
  convolutional_flat = tf.reshape(dropout2,
                                    [-1, numberfeaturemaps])
  print(convolutional_flat)
  
  fullconnectedlayer_w1 = tf.Variable(tf.truncated_normal([numberfeaturemaps, fullconnectedlayer_channel],stddev = 0.01))
  
  
  fullconnectedlayer_b1 = tf.Variable(tf.zeros([fullconnectedlayer_channel]))
  fullconnectedlayer1 = tf.matmul(convolutional_flat, fullconnectedlayer_w1) + fullconnectedlayer_b1

  #second layer full
  fullconnectedlayer_w2 = tf.Variable(tf.truncated_normal([fullconnectedlayer_channel, fullconnectedlayer_channel2], stddev=0.01))
  fullconnectedlayer_b2 = tf.Variable(tf.zeros([fullconnectedlayer_channel2]))
  fullconnectedlayer2 = tf.matmul(fullconnectedlayer1, fullconnectedlayer_w2) + fullconnectedlayer_b2
 

  #layer in classification
  fullconnectedlayer3_w3 = tf.Variable(tf.truncated_normal([fullconnectedlayer_channel2, label], stddev = 0.01))
  fullconnectedlayer3_b3 = tf.Variable(tf.zeros([label]))
  fullconnectedlayer3 = tf.matmul(fullconnectedlayer2, fullconnectedlayer3_w3) + fullconnectedlayer3_b3
  if is_training:
	return fullconnectedlayer3,dropout
  else:
	return fullconnectedlayer3


"""

  put this backpropagation algorithm in the train.py file  
  ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input))
  lr = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
  gradient = tf.train.GradientDescentOptimizer(
        lr).minimize(ce)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(ground_truth_input, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  confusion_matrix = tf.confusion_matrix(tf.argmax(ground_truth_input,1), tf.argmax(ground_truth_input, 1))
  """


