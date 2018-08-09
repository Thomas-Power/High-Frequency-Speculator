import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler
import random
import os
import time

#locate Source_Data
currencies = os.listdir(r"./import/")

logs_path = 'log_relu/'
batch_size = 10
learning_rate = 0.000003
training_epochs = 10000
display_epoch = 1

block_size = 30
max_generations = 5


#create arrays to fill with feed data
Projection_Value = np.zeros([block_size], dtype=float)
batch_x = np.zeros([(batch_size), block_size], dtype=float)
batch_y = np.zeros([(batch_size)], dtype=float)

#Create batch_pool. Data to be fed into neural network
def createBatch(gen):
	
	cur_string = currencies[random.randint(0, (len(currencies)-1))]
	data_files = os.listdir(r"./import/" + cur_string + "/")
	
	num = random.randint(1, (len(data_files)-1))
	
	Source_Path = "./import/" + cur_string + "/" + data_files[num]
	Scale_Path = "./import/" + cur_string + "/" + data_files[num-1]
	#import csv data as dataframe
	Source_Data = pd.read_csv(Source_Path, header=0)
	
	Scale_Data = pd.read_csv(Scale_Path, header=0)
	#Isolate opening value column
	Source_Data = Source_Data["open"]
	Scale_Data = Scale_Data["open"]
	
	#scale data between -1 and 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	Source_Data = np.reshape(Source_Data, (-1, 1))
	scale_mean = Scale_Data.mean()
	scale = [scale_mean * 0.97, scale_mean * 1.03]
	scale = np.reshape(scale, (-1, 1))
	scaler.fit(scale)
	Source_Data = scaler.transform(Source_Data)

	for i in range(batch_size):
		gen_num = gen
		data_range = block_size * gen_num
		max_range = (len(Source_Data) - (12 * max_generations)) - 1
		T = random.randint(data_range, max_range)				# pick random point for T
		T_Vector = Source_Data[T]								# Get vector for T
		Hist_Range = Source_Data[(T-data_range) : T]			# gather 30 points per generation before T as Historical data
		Hist_Price = Hist_Range[0::gen_num]						# isolate 30 points within that range
		Target = Source_Data[T + (gen_num * 6 + ( random.randint(0, 6)))]			# randomly pick T + [6-12] as target value
		for e in range(block_size):
			Projection_Value[e] = Hist_Price[e]

		batch_x[i] = Projection_Value							#projection data and target now defined as batch_pool[i]
		batch_y[i] = float(Target)

	
batch_y = np.reshape(batch_y, (-1, 1))


# Define Neurons per layer

L = 768
M = 256
N = 128
O = 64

X = tf.placeholder(tf.float32, [None, block_size], name='InputData')	 	 # input shape 30
Y = tf.placeholder(tf.float32, [None, 1], name='LabelData')

W1 = tf.Variable(tf.truncated_normal([block_size, L], stddev=0.1)) 	 # random weights for the hidden layer 1
B1 = tf.Variable(tf.ones([L])) 									 	 # bias vector for layer 1
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)  							 # Output from layer 1

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
Y_ = tf.nn.selu(tf.matmul(Y4, W5) + B5, name='Output') 				 # Output from layer 5 

cost_op = tf.reduce_mean(tf.square(Y_ - Y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

tf.summary.scalar("cost", cost_op)
summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
for generation in range(1, 6):
	with tf.Session() as sess:
		sess.run(init_op)																			# run the initializer
		writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) 					# create writer for accuracy logs
		for epoch in range(training_epochs):
			createBatch(generation)																			# prepare data for input																	#separates data by epoch
			for i in range(batch_size):
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})						 		#train data
				_,summary = sess.run([train_op, summary_op], feed_dict={X: batch_x, Y: batch_y})	#produce summary
				writer.add_summary(summary, epoch * batch_size + i)									#record summary
			print("Generation:", generation)
			print("Epoch: ", epoch)
			print("Optimization Finished!")
			print("Accuracy: ", cost_op.eval(feed_dict={X: batch_x, Y: batch_y}))					#print accuracy
		saver.save(sess, './python_models/gen_' + str(generation) + '/trained_model', global_step=1000)									#save model

		builder = tf.saved_model.builder.SavedModelBuilder('./java_models/gen_' + str(generation) + '/model')
		builder.add_meta_graph_and_variables(
		  sess,
		  [tf.saved_model.tag_constants.SERVING]
		)
		builder.save()
		sess.close()

		