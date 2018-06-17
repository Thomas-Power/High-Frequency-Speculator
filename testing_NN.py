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


#locate Source_Data
Source_Path = "./import.csv"

#import csv data as numpy array 
Source_Data = np.loadtxt(open(Source_Path, "rb"), delimiter=",", skiprows=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
Source_Data = np.reshape(Source_Data, (-1, 1))
scaler.fit(Source_Data)
Source_Data = scaler.transform(Source_Data)

#create arrays to fill with feed data
Projection_Value = np.ones([30], dtype=float)
Projection_Time = np.ones([30], dtype=float)
Hist_Source = np.ones([30], dtype=float)
Hist_Time = np.ones([30], dtype=float)
batch_x = np.ones([len(Source_Data)-13, 30], dtype=float)
batch_y = np.ones([len(Source_Data)-13], dtype=float)
preditions = np.ones([len(Source_Data)-13], dtype=float)

#Create batch_pool. Data to be fed into neural network
for i in range(40, (len(Source_Data)-13)):
	T = i
	T_Vector = Source_Data[T]						# Get vector for T
	Hist_Price = Source_Data[(T-30) : T]			# gather 30 points before T as Historical data
	Target = Source_Data[T + 9]						# randomly pick T + [6-12] as target value
	for e in range(30):
		Projection_Value[e] = Hist_Price[e]
		
	batch_x[i] = Projection_Value # projections and target now defined as batch_pool[i]
	batch_y[i] = Target
	
saver = tf.train.import_meta_graph('./temp/trained_model-1000.meta')
with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint('./temp'))
	graph = tf.get_default_graph()
	Y_ = graph.get_tensor_by_name('Output:0')
	X = graph.get_tensor_by_name('InputData:0')
	
	preditions = sess.run(Y_, feed_dict={X: batch_x})

preditions = scaler.inverse_transform(preditions)
batch_y = scaler.inverse_transform(Source_Data)
	
plt.plot(preditions, label="Prediction")
plt.plot(batch_y, label="Actual Data")
plt.legend()
plt.show()