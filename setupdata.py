#Converts "dd/mm/yy H:M" time stamps into Unix value necessary for analysis
import time
import datetime
import pandas as pd
import csv as csv

Source_Path = "./import.csv"
#get csv data as numpy array 
Source_Data = pd.read_csv(Source_Path, header=None)

with open(Source_Path, 'w', newline='') as File:
	writer = csv.writer(File)
	#for i in range(len(Source_Data)):
	for i in range(len(Source_Data)):
		line = [Source_Data[0][i], time.mktime(time.strptime(Source_Data[1][i], "%d/%m/%Y %H:%M"))]
		writer.writerow(line)
		#print(time.mktime(time.strptime(Source_Data[1][i], "%d/%m/%Y %H:%M")))