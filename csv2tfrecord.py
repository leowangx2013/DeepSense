import os

import tensorflow as tf
import numpy as np
import random

TF_RECORD_PATH = r'A:\Research\Accelerometer\DeepSense_TensorRecord\DeepSense\tensor_records'
CSV_PATH = r'A:\Research\Accelerometer\AccelerometerSpeechRecognition\DeepSense\MyData'

TRAINING_RATIO = 0.8
TESTING_RATIO = 0.2

ONE_HOT_LABEL_LEN = 10

def csv_to_example(fname):
    print(fname)
    text = np.loadtxt(fname, delimiter=',')
    features = text[:-10]
    label = text[-10:]
    print("features len = {}".format(len(features)))
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        'example': tf.train.Feature(float_list=tf.train.FloatList(value=features))
    }))

    return example


data_files = []
for f in os.listdir(CSV_PATH):
	if f.endswith(".csv"):
		data_files.append(f)
random.shuffle(data_files)

train_files = data_files[:int(len(data_files)*TRAINING_RATIO)]

test_files = data_files[int(len(data_files)*TRAINING_RATIO):]

writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'train.tfrecord'))

for f in train_files:
    f_path = os.path.join(CSV_PATH, f)
    example = csv_to_example(f_path)
    writer.write(example.SerializeToString())
writer.close()


writer = tf.python_io.TFRecordWriter(os.path.join(TF_RECORD_PATH, 'test.tfrecord'))

for f in test_files:
    f_path = os.path.join(CSV_PATH, f)
    example = csv_to_example(f_path)
    writer.write(example.SerializeToString())
writer.close()