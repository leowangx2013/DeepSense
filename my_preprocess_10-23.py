import json
import os
import glob
import numpy as np
import csv
from scipy import signal

ACCELEROMETER_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_accelerometer_clips_10_22"
GYROSCOPE_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_gyroscope_clips_10_22"
# GYROSCOPE_FILE_PATH = "A:\Research\Accelerometer\Accelerometer+Gyroscope\lg_accelerometer_clips"

OUTPUT_PATH = "A:\Research\Accelerometer\AccelerometerSpeechRecognition\DeepSense\MyData_10-22"

# SPECTURM_SAMPLE_NUM = 25
SPECTURM_SAMPLE_NUM = 8

LABELS = range(0, 10)

def load_data_under_dir(data_dir):
	filenames_under_dir = [name for name in glob.glob("{}\*.txt".format(data_dir))]
	data = []
	for filename in filenames_under_dir:
		print(filename)
		with open(filename, "r") as file:
			data.append(json.load(file))
	# print("data[0]: {}".format(data[0]))
	# exit()
	return data

accelerometer_data = load_data_under_dir(ACCELEROMETER_FILE_PATH)
gyroscope_data = load_data_under_dir(GYROSCOPE_FILE_PATH)

def split_into_chunks(data, chunk_size):
	data -= np.array(data).mean(axis=0, keepdims=True)

	for i in range(0, len(data), chunk_size):
		yield data[i: i+chunk_size]

def preprocess_one_label(accelerometer_data, gyroscope_data, segment_length, means):
	# print("accelerometer_data.shape: {}".format(np.array(accelerometer_data).shape))
	X = []
	print("accelerometer_data.shape: {}".format(accelerometer_data.shape))
	print("accelerometer_data[0]: {}".format(accelerometer_data[0])) # Same

	acc_Xs = accelerometer_data[:,:,0]
	acc_Ys = accelerometer_data[:,:,1]
	acc_Zs = accelerometer_data[:,:,2]
	# print("acc_Xs.shape = {}".format(np.array(acc_Xs).shape))
	acc_Xs = acc_Xs - means[0][0]
	acc_Ys = acc_Ys - means[0][1]
	acc_Zs = acc_Zs - means[0][2]
	# print("acc_Xs[0]: {}".format(acc_Xs[0]))

	gyro_Xs = gyroscope_data[:,:,0]
	gyro_Ys = gyroscope_data[:,:,1]
	gyro_Zs = gyroscope_data[:,:,2]

	gyro_Xs = gyro_Xs - means[1][0]
	gyro_Ys = gyro_Ys - means[1][1]
	gyro_Zs = gyro_Zs - means[1][2]

	#_, _, acc_Zx = signal.stft(acc_Xs[self.current_idx].astype(np.float), nperseg = segment_length)
	#_, _, acc_Zy = signal.stft(acc_Ys[self.current_idx].astype(np.float), nperseg = segment_length)
	#_, _, acc_Zz = signal.stft(acc_Zs[self.current_idx].astype(np.float), nperseg = segment_length)
	for i in range(len(acc_Xs)):
		_, _, acc_Zx = signal.stft(acc_Xs[i].astype(np.float), nperseg = segment_length)
		_, _, acc_Zy = signal.stft(acc_Ys[i].astype(np.float), nperseg = segment_length)
		_, _, acc_Zz = signal.stft(acc_Zs[i].astype(np.float), nperseg = segment_length)
		# print("first acc_Zx: {}".format(acc_Zx))
		# exit()
		# print("gyro_data.shape = {}".format(np.array(self.gyro_data).shape))
		# print("gyro_Xs.shape: {}".format(np.array(gyro_Xs).shape))
		#_, _, gyro_Zx = signal.stft(gyro_Xs[self.current_idx].astype(np.float), nperseg = segment_length)
		#_, _, gyro_Zy = signal.stft(gyro_Ys[self.current_idx].astype(np.float), nperseg = segment_length)
		#_, _, gyro_Zz = signal.stft(gyro_Zs[self.current_idx].astype(np.float), nperseg = segment_length)
		_, _, gyro_Zx = signal.stft(gyro_Xs[i].astype(np.float), nperseg = segment_length)
		_, _, gyro_Zy = signal.stft(gyro_Ys[i].astype(np.float), nperseg = segment_length)
		_, _, gyro_Zz = signal.stft(gyro_Zs[i].astype(np.float), nperseg = segment_length)

		# Zx = Zx.transpose((1, 0))
		# Zy = Zy.transpose((1, 0))
		# Zz = Zz.transpose((1, 0))

		'''Combine 3 axis'''
		X.append([[acc_Zx[j], acc_Zy[j], acc_Zz[j], gyro_Zx[j], gyro_Zy[j], gyro_Zz[j]] for j in range(len(acc_Zx))])
		
	print("X.shape: {}".format(np.array(X).shape))

	return X

	##############################

	# for (acc_clip, gyro_clip) in zip(accelerometer_data, gyroscope_data):
	# 	X_T = []
		
	# 	print("accelerometer_data.shape: {}".format(np.array(accelerometer_data).shape))

	# 	acc_chunks = list(split_into_chunks(acc_clip, SPECTURM_SAMPLE_NUM))
	# 	gyro_chunks = list(split_into_chunks(gyro_clip, SPECTURM_SAMPLE_NUM))

	# 	for (acc_c, gyro_c) in zip(acc_chunks, gyro_chunks):
	# 		# print("acc_c.shape: {}".format(np.transpose(acc_c).shape))
	# 		t_acc_c = np.transpose(acc_c)
	# 		# print("t_acc_c.shape = {}".format(t_acc_c.shape))
	# 		# acc_x_fft_result = np.fft.fft(t_acc_c[0]-np.mean(t_acc_c[0]))
	# 		# acc_y_fft_result = np.fft.fft(t_acc_c[1]-np.mean(t_acc_c[1]))
	# 		# acc_z_fft_result = np.fft.fft(t_acc_c[2]-np.mean(t_acc_c[2]))
	# 		acc_x_fft_result = np.fft.fft(t_acc_c[0])
	# 		acc_y_fft_result = np.fft.fft(t_acc_c[1])
	# 		acc_z_fft_result = np.fft.fft(t_acc_c[2])

	# 		acc_fft_result = [[x.real, x.imag, y.real, y.imag, z.real, z.imag] for 
	# 			(x, y, z) in zip(acc_x_fft_result, acc_y_fft_result, acc_z_fft_result)]		
	# 		# acc_fft_result = [[np.abs(x), np.abs(y), np.abs(z)] for 
	# 		#  	(x, y, z) in zip(acc_x_fft_result, acc_y_fft_result, acc_z_fft_result)]		

	# 		t_gyro_c = np.transpose(gyro_c)
	# 		# gyro_x_fft_result = np.fft.fft(t_gyro_c[0]-np.mean(t_gyro_c[0]))
	# 		# gyro_y_fft_result = np.fft.fft(t_gyro_c[1]-np.mean(t_gyro_c[1]))
	# 		# gyro_z_fft_result = np.fft.fft(t_gyro_c[2]-np.mean(t_gyro_c[2]))
	# 		gyro_x_fft_result = np.fft.fft(t_gyro_c[0])
	# 		# print("gyro_x_fft_result.shape: {}".format(gyro_x_fft_result.shape))
	# 		gyro_y_fft_result = np.fft.fft(t_gyro_c[1])
	# 		gyro_z_fft_result = np.fft.fft(t_gyro_c[2])

	# 		gyro_fft_result = [[x.real, x.imag, y.real, y.imag, z.real, z.imag] for 
	# 			(x, y, z) in zip(gyro_x_fft_result, gyro_y_fft_result, gyro_z_fft_result)]
	# 		# gyro_fft_result = [[np.abs(x), np.abs(y), np.abs(z)] for 
	# 		#  	(x, y, z) in zip(gyro_x_fft_result, gyro_y_fft_result, gyro_z_fft_result)]

	# 		combined_fft_result = [[a, g] for (a, g) in zip(acc_fft_result, gyro_fft_result)]
	# 		X_T.append(combined_fft_result)

	# 	X.append(X_T)
	# # print("X.shape: {}".format(np.array(X).shape))
	# return X

counter = 0

accelerometer_data = np.array(accelerometer_data)
gyroscope_data = np.array(gyroscope_data)

merged_accelerometer_data = []
merged_gyroscope_data = []

for label in LABELS:
	merged_accelerometer_data += accelerometer_data[label]
	merged_gyroscope_data += gyroscope_data[label]

merged_accelerometer_data = np.array(merged_accelerometer_data)
print("merged_accelerometer_data.shape: {}".format(merged_accelerometer_data.shape))
merged_gyroscope_data = np.array(merged_gyroscope_data)

x_mean_acc = np.mean(merged_accelerometer_data[:,:,0])
y_mean_acc = np.mean(merged_accelerometer_data[:,:,1])
z_mean_acc = np.mean(merged_accelerometer_data[:,:,2])

x_mean_gyro = np.mean(merged_gyroscope_data[:,:,0])
y_mean_gyro = np.mean(merged_gyroscope_data[:,:,1])
z_mean_gyro = np.mean(merged_gyroscope_data[:,:,2])

means = [[x_mean_acc, y_mean_acc, z_mean_acc], [x_mean_gyro, y_mean_gyro, z_mean_gyro]]

print("means: {}".format(means))

for label in LABELS:
	print("label: {}".format(label))
	print("accelerometer_data[label][0]: {}".format(accelerometer_data[label][0]))
	X = preprocess_one_label(np.array(accelerometer_data[label]), np.array(gyroscope_data[label]), SPECTURM_SAMPLE_NUM, means)
	for x in X:
		# print("x.shape: ", np.array(x).shape)
		with open(os.path.join(OUTPUT_PATH, "train_{}.csv".format(counter)), "w", newline='') as file:
			writer = csv.writer(file)
			one_hot_label = np.zeros(len(LABELS))
			one_hot_label[label] = 1
			one_hot_label = [str(i) for i in one_hot_label]
			writer.writerow(np.concatenate((np.array(x).flatten(), one_hot_label)))
		counter += 1	
