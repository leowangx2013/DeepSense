import json
import os
import glob
import numpy as np
import csv

ACCELEROMETER_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_accelerometer_clips"
GYROSCOPE_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_gyroscope_clips"
OUTPUT_PATH = "A:\Research\Accelerometer\AccelerometerSpeechRecognition\DeepSense\MyData"

INTERVALS_NUM = 10
SPECTURM_SAMPLE_NUM = 25

LABELS = range(0, 10)

def load_data_under_dir(data_dir):
	filenames_under_dir = [name for name in glob.glob("{}\*.txt".format(data_dir))]
	data = []
	for filename in filenames_under_dir:
		with open(filename, "r") as file:
			data.append(json.load(file))
	return data

accelerometer_data = load_data_under_dir(ACCELEROMETER_FILE_PATH)
gyroscope_data = load_data_under_dir(GYROSCOPE_FILE_PATH)

def split_into_chunks(data, chunk_size):
	for i in range(0, len(data), chunk_size):
		yield data[i: i+chunk_size]

def preprocess_one_label(accelerometer_data, gyroscope_data):
	X = []
	for (acc_clip, gyro_clip) in zip(accelerometer_data, gyroscope_data):
		X_T = []

		acc_chunks = list(split_into_chunks(acc_clip, SPECTURM_SAMPLE_NUM))
		gyro_chunks = list(split_into_chunks(gyro_clip, SPECTURM_SAMPLE_NUM))
		
		for (acc_c, gyro_c) in zip(acc_chunks, gyro_chunks):
			# print("acc_c.shape: {}".format(np.transpose(acc_c).shape))
			acc_x_fft_result = np.fft.fft(np.transpose(acc_c)[0])
			acc_y_fft_result = np.fft.fft(np.transpose(acc_c)[1])
			acc_z_fft_result = np.fft.fft(np.transpose(acc_c)[2])

			acc_fft_result = [(x.real, x.imag, y.real, y.imag, z.real, z.imag) for 
				(x, y, z) in zip(acc_x_fft_result, acc_y_fft_result, acc_z_fft_result)]		

			gyro_x_fft_result = np.fft.fft(np.transpose(gyro_c)[0])
			gyro_y_fft_result = np.fft.fft(np.transpose(gyro_c)[1])
			gyro_z_fft_result = np.fft.fft(np.transpose(gyro_c)[2])

			gyro_fft_result = [(x.real, x.imag, y.real, y.imag, z.real, z.imag) for 
				(x, y, z) in zip(gyro_x_fft_result, gyro_y_fft_result, gyro_z_fft_result)]
			
			combined_fft_result = [acc_fft_result, gyro_fft_result]
			X_T.append(combined_fft_result)

		X.append(X_T)
	print("X.shape: {}".format(np.array(X).shape))
	return X

counter = 0
for label in LABELS:
	X = preprocess_one_label(accelerometer_data[label], gyroscope_data[label])
	for x in X:
		with open(os.path.join(OUTPUT_PATH, "train_{}.csv".format(counter)), "w") as file:
			writer = csv.writer(file)
			one_hot_label = np.zeros(len(LABELS))
			one_hot_label[label] = 1
			writer.writerow(np.concatenate((np.array(x).flatten(), one_hot_label)))

			counter+=1