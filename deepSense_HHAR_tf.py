import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

MODE = "TESTING"	
CHECKPOINT_PATH = "A:\Research\Accelerometer\AccelerometerSpeechRecognition\DeepSense\checkpoints_tf_record\deepsense_model"
layers = tf.contrib.layers 

# SEPCTURAL_SAMPLES = 10
# SEPCTURAL_SAMPLES = 25
SEPCTURAL_SAMPLES = 10

# FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2

CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
# OUT_DIM = 6#len(idDict)
OUT_DIM = 10
# WIDE = 20
# WIDE = 10
WIDE = 25
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(2004*0.2/64)
TF_RECORD_PATH = "A:\Research\Accelerometer\DeepSense_TensorRecord\DeepSense\\tensor_records"

###### Import training data
def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)

	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + OUT_DIM)]
	fileData = tf.decode_csv(value, record_defaults=defaultVal)

	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])
	print("features.shape: {}".format(features.get_shape().as_list()))

	labels = fileData[WIDE*FEATURE_DIM:]
	print("labels.shape: {}".format(np.array(labels).shape))
	return features, labels

def read_tf_record(tfrec_path):
	print("tfrec_path: {}".format(tfrec_path))
	filename_queue = tf.train.string_input_producer([tfrec_path])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	example = tf.parse_single_example(serialized_example,
									   features={
										   'label': tf.FixedLenFeature([OUT_DIM], tf.float32),
										   'example': tf.FixedLenFeature([WIDE*FEATURE_DIM], tf.float32),
									   })
	features = tf.reshape(example['example'], shape=[WIDE, FEATURE_DIM])
	label = example['label']
	return features, label

def input_pipeline_tf_record(tfrec_path, batch_size, shuffle_sample=True, num_epochs=None):
	example, label = read_tf_record(tfrec_path)

	print("example.shape: {}".format(example.get_shape().as_list()))
	print("label.shape: {}".format(label.get_shape().as_list()))

	min_after_dequeue = 1000  # int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	print("example_batch.shape: {}, label_batch.shape: {}".format(
		example_batch.get_shape().as_list(), label_batch.get_shape().as_list()))
	return example_batch, label_batch


def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	print("example.shape: {}".format(example.get_shape().as_list()))
	print("label.shape: {}".format(np.array(label).shape))
	
	min_after_dequeue = 1000 #int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)

	print("example_batch.shape: {}, label_batch.shape: {}".format(
		example_batch.get_shape().as_list(), label_batch.get_shape().as_list()))

	return example_batch, label_batch

######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,  
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True, 
# 			updates_collections=None, scope=scope),  
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True)) 

def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)

		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		# sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		sensor_inputs = tf.reshape(inputs, [-1, WIDE, SEPCTURAL_SAMPLES, 2, 6])
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=3)
		acc_inputs = tf.squeeze(acc_inputs, axis=3)
		gyro_inputs = tf.squeeze(gyro_inputs, axis=3)
		print("acc_inputs.shape: {}".format(acc_inputs.get_shape().as_list()))
		print("gyro_inputs.shape: {}".format(gyro_inputs.get_shape().as_list()))

		acc_conv1 = layers.convolution2d(acc_inputs, CONV_NUM, kernel_size=[1, CONV_LEN],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv1')
		acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
		acc_conv1 = tf.nn.relu(acc_conv1)
		acc_conv1_shape = acc_conv1.get_shape().as_list()
		acc_conv1 = layers.dropout(acc_conv1, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[acc_conv1_shape[0], 1, 1, acc_conv1_shape[3]], scope='acc_dropout1')

		acc_conv2 = layers.convolution2d(acc_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv2')
		acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
		acc_conv2 = tf.nn.relu(acc_conv2)
		acc_conv2_shape = acc_conv2.get_shape().as_list()
		acc_conv2 = layers.dropout(acc_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[acc_conv2_shape[0], 1, 1, acc_conv2_shape[3]], scope='acc_dropout2')

		acc_conv3 = layers.convolution2d(acc_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv3')
		acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
		acc_conv3 = tf.nn.relu(acc_conv3)
		acc_conv3_shape = acc_conv3.get_shape().as_list()
		acc_conv_out = tf.reshape(acc_conv3, [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2],acc_conv3_shape[3]])


		gyro_conv1 = layers.convolution2d(gyro_inputs, CONV_NUM, kernel_size=[1, CONV_LEN],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv1')
		gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
		gyro_conv1 = tf.nn.relu(gyro_conv1)
		gyro_conv1_shape = gyro_conv1.get_shape().as_list()
		gyro_conv1 = layers.dropout(gyro_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[gyro_conv1_shape[0], 1, 1, gyro_conv1_shape[3]], scope='gyro_dropout1')

		gyro_conv2 = layers.convolution2d(gyro_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv2')
		gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
		gyro_conv2 = tf.nn.relu(gyro_conv2)
		gyro_conv2_shape = gyro_conv2.get_shape().as_list()
		gyro_conv2 = layers.dropout(gyro_conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[gyro_conv2_shape[0], 1, 1, gyro_conv2_shape[3]], scope='gyro_dropout2')

		gyro_conv3 = layers.convolution2d(gyro_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', data_format='NHWC', scope='gyro_conv3')
		gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
		gyro_conv3 = tf.nn.relu(gyro_conv3)
		gyro_conv3_shape = gyro_conv3.get_shape().as_list()
		gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])	


		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()	
		sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
			noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], scope='sensor_dropout_in')

		sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv1')
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.get_shape().as_list()
		sensor_conv1 = layers.dropout(sensor_conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]], scope='sensor_dropout1')

		sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN2],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv2')
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.get_shape().as_list()
		sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]], scope='sensor_dropout2')

		sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN3],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv3')
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.get_shape().as_list()
		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])

		# gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
		lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(INTER_DIM)
		if train:
			# gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)
			lstm_cell1 = tf.contrib.rnn.DropoutWrapper(lstm_cell1, output_keep_prob=0.5)

		# gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
		lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(INTER_DIM)

		if train:
			lstm_cell2 = tf.contrib.rnn.DropoutWrapper(lstm_cell2, output_keep_prob=0.5)

		# cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])

		init_state = cell.zero_state(BATCH_SIZE, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

		sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		avg_cell_out = sum_cell_out/avgNum

		logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits

def count_trainable_parameters():
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		# print(shape)
		# print(len(shape))
		variable_parameters = 1
		for dim in shape:
			# print(dim)
			variable_parameters *= dim.value
		# print(variable_parameters)
		total_parameters += variable_parameters
	print("=============================================")
	print("total_parameters: {}".format(total_parameters))
	print("=============================================")

csvFileList = []
csvDataFolder1 = "./MyData"
orgCsvFileList = os.listdir(csvDataFolder1)
for csvFile in orgCsvFileList:
	if csvFile.endswith('.csv'):
		csvFileList.append(os.path.join(csvDataFolder1, csvFile))

# csvEvalFileList = []
# csvDataFolder2 = os.path.join('sepHARData_'+select, "eval")
# orgCsvFileList = os.listdir(csvDataFolder2)
# for csvFile in orgCsvFileList:
# 	if csvFile.endswith('.csv'):
# 		csvEvalFileList.append(os.path.join(csvDataFolder2, csvFile))


# batch_feature, batch_label = input_pipeline(csvFileList, BATCH_SIZE)

# batch_eval_feature, batch_eval_label = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)
batch_feature, batch_label = input_pipeline_tf_record(os.path.join(TF_RECORD_PATH, 'train.tfrecord'), BATCH_SIZE)
batch_eval_feature, batch_eval_label = input_pipeline_tf_record(os.path.join(TF_RECORD_PATH, 'test.tfrecord'), BATCH_SIZE, shuffle_sample=False)

# train_status = tf.placeholder(tf.bool)
# trainX = tf.cond(train_status, lambda: tf.identity(batch_feature), lambda: tf.identity(batch_eval_feature))
# trainY = tf.cond(train_status, lambda: tf.identity(batch_label), lambda: tf.identity(batch_eval_label))

global_step = tf.Variable(0, trainable=False)

# logits = deepSense(trainX, train_status, name='deepSense')
logits = deepSense(batch_feature, True, name='deepSense')
predict = tf.argmax(logits, axis=1)
# batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trainY)
batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
loss = tf.reduce_mean(batchLoss)

logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
predict_eval = tf.argmax(logits_eval, axis=1)
# loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

t_vars = tf.trainable_variables()

regularizers = 0.
for var in t_vars:
	regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers

# optimizer = tf.train.RMSPropOptimizer(0.001)
# gvs = optimizer.compute_gradients(loss, var_list=t_vars)
# capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
# optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)

optimizer = tf.train.AdamOptimizer(
		learning_rate=1e-4, 
		beta1=0.9,
		beta2=0.999,
	).minimize(loss, var_list=t_vars)

# batch = tf.Variable(0, dtype=data_type())

# learning_rate = tf.train.exponential_decay(
# 	0.01,                # Base learning rate.
# 	batch * BATCH_SIZE,  # Current index into the dataset.
# 	100,          	     # Decay step.
# 	0.95,                # Decay rate.
# 	staircase=True)

# optimizer = tf.train.MomentumOptimizer(learning_rate,
#         0.9).minimize(loss, global_step=batch)

count_trainable_parameters()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

if MODE == "TRAINING":

	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for iteration in range(TOTAL_ITER_NUM):

			# _, lossV, _trainY, _predict = sess.run([optimizer, loss, trainY, predict], feed_dict = {
			# 	train_status: True
			# 	})
			_, lossV, _trainY, _predict = sess.run([optimizer, loss, batch_label, predict])
			
			_label = np.argmax(_trainY, axis=1)
			_accuracy = np.mean(_label == _predict)
			# plot.plot('train cross entropy', lossV)
			# plot.plot('train accuracy', _accuracy)

			print("iteration: {}, accuracy = {}, loss = {}".format(iteration, _accuracy, lossV))
			print("prediction = {}".format(_predict))
			print("ground truth = {}".format(_label))

			# saver.save(sess, 'A:\Research\Accelerometer\AccelerometerSpeechRecognition\DeepSense\checkpoints\deepsense_model')
			saver.save(sess, CHECKPOINT_PATH)

			if iteration % 5 == 4:
				dev_accuracy = []
				dev_cross_entropy = []
				for eval_idx in range(EVAL_ITER_NUM):
					# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
					eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])
					_label = np.argmax(_trainY, axis=1)
					_accuracy = np.mean(_label == _predict)
					dev_accuracy.append(_accuracy)
					dev_cross_entropy.append(eval_loss_v)

				print("iteration: {}, testing accuracy = {}, testing loss = {}".format(
					iteration, np.mean(dev_accuracy), np.mean(dev_cross_entropy)))
				print("")
else:
	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		saver.restore(sess, CHECKPOINT_PATH)

		dev_accuracy = []
		dev_cross_entropy = []
		for eval_idx in range(EVAL_ITER_NUM):
			# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
			eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])
			_label = np.argmax(_trainY, axis=1)
			_accuracy = np.mean(_label == _predict)
			dev_accuracy.append(_accuracy)
			dev_cross_entropy.append(eval_loss_v)

		print("testing accuracy = {}, testing loss = {}".format(
			np.mean(dev_accuracy), np.mean(dev_cross_entropy)))
		print("prediction = {}".format(_predict))
		print("ground truth = {}".format(_label))