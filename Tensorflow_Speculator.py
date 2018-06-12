import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler
import random
import os
import time

logs_path = 'log_relu/'
batch_count = 100
learning_rate = 0.0003
training_epochs = 10
display_epoch = 1

#locate Source_Data
Source_Path = "./import.csv"

#import csv data as numpy array 
Source_Data = np.loadtxt(open(Source_Path, "rb"), delimiter=",", skiprows=1)
		
#scale data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
Source_Data = np.reshape(Source_Data, (-1, 1))
scaler.fit(Source_Data)
Source_Data = scaler.transform(Source_Data)


#create arrays to fill with feed data
Projection_Value = np.ones([30], dtype=float)
Hist_Source = np.ones([30], dtype=float)
batch_x = np.ones([(batch_count*training_epochs), 30], dtype=float)
batch_y = np.ones([(batch_count*training_epochs)], dtype=float)

#Create batch_pool. Data to be fed into neural network
def createBatch():
	for epoch in range(training_epochs):
		f = batch_count * epoch 
		for i in range(batch_count):
			T = random.randint(30, len(Source_Data)-13)				# pick random point for T
			T_Vector = Source_Data[T]								# Get vector for T
			Hist_Price = Source_Data[(T-30) : T]					# gather 30 points before T as Historical data
			Target = Source_Data[T + random.randint(6, 12)]			# randomly pick T + [6-12] as target value
			for e in range(30):
				Projection_Value[e] = Hist_Price[e]

			batch_x[i+f] = Projection_Value							#projection data and target now defined as batch_pool[i]
			batch_y[i+f] = float(Target)

		
batch_y = np.reshape(batch_y, (-1, 1))
# Define Neurons per layer
L = 12
M = 6
N = 4
O = 2

X = tf.placeholder(tf.float32, [None, 30], name='InputData')	 # input shape 30*2 value change and time distance
Y = tf.placeholder(tf.float32, [None, 1], name='InputData')

W1 = tf.Variable(tf.truncated_normal([30, L], stddev=0.1)) 			 # random weights for the hidden layer 1
B1 = tf.Variable(tf.ones([L])) 									 # bias vector for layer 1
Y1 = tf.nn.softsign(tf.matmul(X, W1) + B1)  							 # Output from layer 1

W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1)) 			 # random weights for hidden layer 2
B2 = tf.Variable(tf.ones([M])) 										 # bias vector for layer 2
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2) 							 # Output from layer 2

W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))			 # random weights for hidden layer 3
B3 = tf.Variable(tf.ones([N])) 										 # bias vector for layer 3
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3) 							 # Output from layer 3

W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1)) 			 # random weights for hidden layer 4
B4 = tf.Variable(tf.ones([O]))										 # bias vector for layer 4
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4) 							 # Ouput from layer 4

W5 = tf.Variable(tf.truncated_normal([O, 1], stddev=0.1)) 			 # random weights for hidden layer 5
B5 = tf.Variable(tf.ones([1])) 										 # bias vector for layer 5
Y_ = tf.nn.relu(tf.matmul(Y4, W5) + B5, name='Output') 							 # Output from layer 5 

cost_op = tf.reduce_mean(tf.square(Y_ - Y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

tf.summary.scalar("cost", cost_op)
summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init_op)																			# run the initializer
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) 					# create writer for accuracy logs
	createBatch()																				# prepare data for input

	for epoch in range(training_epochs):
		f = batch_count * epoch																	#separates data by epoch
		for i in range(batch_count):
			sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})						 		#train data
			_,summary = sess.run([train_op, summary_op], feed_dict={X: batch_x, Y: batch_y})	#produce summary
			writer.add_summary(summary, epoch * batch_count + (i+f))							#record summary
		#print(scaler.inverse_transform(sess.run(Y_, feed_dict={X: batch_x})))
		print("Epoch: ", epoch)
		print("Optimization Finished!")
		print("Accuracy: ", cost_op.eval(feed_dict={X: batch_x, Y: batch_y}))				#print accuracy
	saver.save(sess, './temp/trained_model', global_step=1000)		
	#save model
	