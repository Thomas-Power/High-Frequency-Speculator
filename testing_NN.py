import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler
import random
import os
import time

batch_size = 30
max_generations = 5

gen_num = random.randint(0, (max_generations+1))

#locate Source_Data
currencies = os.listdir(r"./TestFiles/")
cur_string = currencies[random.randint(0, (len(currencies)-1))]
data_files = os.listdir(r"./TestFiles/" + cur_string + "/")
	
Source_Path = "./TestFiles/" + cur_string + "/" + data_files[random.randint(0, (len(data_files)-1))]


#import csv data as array 
Source_Data = pd.read_csv(Source_Path, header=0)

Source_Data = Source_Data["open"]

scaler = MinMaxScaler(feature_range=(-1, 1))
Source_Data = np.reshape(Source_Data, (-1, 1))
scaler.fit(Source_Data)
Source_Data = scaler.transform(Source_Data)
data_range = int(batch_size * gen_num)
max_range = int(len(Source_Data) - (9 * max_generations) - 1)
target_point = 9 * gen_num


#create arrays to fill with feed data
Projection_Value = np.zeros([batch_size], dtype=float)
Projection_Time = np.zeros([batch_size], dtype=float)
Hist_Source = np.zeros([batch_size], dtype=float)
Hist_Time = np.zeros([batch_size], dtype=float)
batch_x = np.zeros([max_range, batch_size], dtype=float)
batch_y = np.zeros([max_range], dtype=float)
batch_gen = np.zeros([max_range], dtype=float)

preditions = np.zeros([max_range], dtype=float)

#Create batch_pool. Data to be fed into neural network
for i in range(data_range, max_range):
		
	T = i
	T_Vector = Source_Data[T]								# Get vector for T
	Hist_Range = Source_Data[(T-data_range) : T]			# gather 30 points per generation before T as Historical data
	Hist_Price = Hist_Range[0::gen_num]						# isolate 30 points within that range
	Target = Source_Data[T + target_point]					# get target value for comparison
	for e in range(batch_size):
		Projection_Value[e] = Hist_Price[e]
		
	batch_x[i] = Projection_Value # projections and target now defined as batch_pool[i]
	batch_y[i] = Target


	
saver = tf.train.import_meta_graph('./python_models/gen_' + str(gen_num) + '/trained_model-1000.meta')
with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint('./python_models/gen_' + str(gen_num)))
	graph = tf.get_default_graph()
	Y_ = graph.get_tensor_by_name('Output:0')
	X = graph.get_tensor_by_name('InputData:0')
	preditions = sess.run(Y_, feed_dict={X: batch_x})

preditions = scaler.inverse_transform(preditions)
batch_y = scaler.inverse_transform(Source_Data)
	
plt.plot(preditions, label="Prediction")
plt.plot(batch_y, label="Actual Data")
print(str(gen_num*6) + "-" + str((gen_num*6)+6) + "min projection for " + Source_Path[12:15])
plt.title(str(gen_num*6) + "-" + str((gen_num*6)+6) + "min projection for " + Source_Path[12:15]) 
plt.legend()
plt.show()