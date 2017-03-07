# ML 2017 hw1 

import numpy as np
import sys

# read from csv and trim all chinese words
train = []
with open(sys.argv[1], 'rb') as f:
	data = f.read().splitlines()
	for line in data[1:]: # trim the first data which is header
		line = [x.replace("\'", "") for x in str(line).split(',')[3:]]
		train.append(line)

# 11-th data is string
ITERATION = 100
DATA_SIZE = int(len(train) / 18)
FEATURE = [0] * 18
ETA = 0.0001

HOURS_OF_DAY = 24

SF = 9 # PM 2.5 # selected feature
PERIOD = 9
w = np.array([1] * (PERIOD + 1)) # 9 hr pm2.5 feature and constant

all_Ein = []

for i in range(ITERATION):

	for d in range(DATA_SIZE):

		print(d, "data")

		data = [float(x) for x in train[18 * d + SF]]
		print("PM 2.5 data:", data)

		this_Ein = []
		for start in range(HOURS_OF_DAY - PERIOD - 1):
			
			fetch_data = data[start : start + PERIOD] # fetch 9 hr of data
			fetch_data.append(1.0) # append constant
			np_data = np.array(fetch_data)

			yh = data[start + PERIOD] # right answer => y hat
			
			# we divide the dot result by (PERIOD + 1) to make it a ratio of every features
			gradient_of_error = 2. * ( yh - np.dot(np_data, w) / (PERIOD + 1)) * (-1.) * np_data

			# compute new w
			w = w - ETA * gradient_of_error
			print("w =", ["{:.3f}".format(x) for x in w], end=", ")

			Ein = ( yh - np.dot(np_data, w) / (PERIOD + 1)) ** 2
			print("Ein =", Ein)

			this_Ein.append(Ein)

		average_this_Ein = sum(this_Ein)/(HOURS_OF_DAY - PERIOD - 1)
		print("this_Ein =", average_this_Ein)
		
		all_Ein.append(average_this_Ein)

print("average Ein = ", sum(all_Ein)/(DATA_SIZE * ITERATION) )
