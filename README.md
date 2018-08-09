## Neural-Net-Speculator
Neural-network designed to provide accurate ticker value projections of prices within specified timeframes.

#### Written via: 
Python

#### Featured libraries: 
TensorFlow, Numpy, Pandas, Matplotlib

#### To run: 
To run test demo simply launch training_NN.py with Tensorflow and appropriate libraries installed.

#### Design process:
I was tasked with inventing a procedure by which our trading system could project prices into the future in order to keep track of moving trends on a minute by minute basis. I seen early on machine learning could be an excellent avenue of exploration. Upon reviewing work by others and my own early tests I found promising results to develope further. 
Machine Learning is an ever evolving process of design and many iterations and tests were run until reaching the current version. Problems encountered early on was establishing a uniform means of normalizing input values (the activation functions available in Tensorflow as a rule work best with figures scaled between 0 and 1 or -1 and 1). 
Heuristic analysis of historical deviation was found to be the most reliable approach. On analyzing the data of the following commodity it was found that a deviation of 3% from the latest 6-hour mean produced reliable and coverage.
The following graph was used in analyzing maximum mean price deviation:

![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/XBT_mean.png)

The model itself is a feed-forward dense neural network. This description details that data is forward moving through each layer of nodes from the input layer through the hidden layers to the eventual output. From reviewing others working in the area ReLU activation functions are successful in producing results for this form of market data but on my own analysis of results a SeLU function placed at the output layer produced far more accurate results in tests though inferior returns placed elsewhere in the model.

#### Perfomance on test data snapshots:
(Test data was naturally seperated from training data)

![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/1.png)
![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/2.png)
![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/3.png)
![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/4.png)
![alt text](https://raw.githubusercontent.com/Thomas-Power/High-Frequency-Speculator/master/Test%20Graphs/5.png)

*Note flatline at beginning of graph represents minimum necessary input range
