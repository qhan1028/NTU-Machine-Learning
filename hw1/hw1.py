# ML 2017 hw1 

import numpy as np
import sys

# read from csv and trim all chinese words
train = []
with open(sys.argv[1], 'rb') as f:
	data = f.read().splitlines()
	i = 0
	for line in data[1:]: # trim the first data which is header
		line = [x.replace("\'", "") for x in str(line).split(',')[3:]]
		if i % 18 != 10:
			line = [float(x) for x in line]
		train.append(line)
		i += 1

# 11-th data is string
ITERATION = 10
DATA_SIZE = int(len(train) / 18)
ETA = 0.0001

HOURS_OF_DAY = 24

SF = [9] # selected feature
PERIOD = 9
w = [ [1.0] * (PERIOD + 1) ] * len(SF) # 9 hr pm2.5 feature and constant
w = np.array(w)
size_w = float(len(w) * len(w[0]))

all_Ein = []

def filter_data(SF, d):
	
	result = []
	for sf in SF:
		
		result.append(train[d * 18 + sf])
	
	return result

def filter_hours(data, start):

	result = []
	for l in range(len(data)):
	
		# fetch data in period and add constant
		result.append(data[l][start : start + PERIOD] + [1.0])
	
	return result

def predict(data, w):
	
	mul = data * w
	return np.sum(mul) / size_w

for i in range(ITERATION):

	for d in range(DATA_SIZE):


		data = filter_data(SF, d)

		this_Ein = []
		for start in range(HOURS_OF_DAY - PERIOD - 1):
			
			fetch_data = filter_hours(data, start) # fetch hrs of data + constant
			np_data = np.array(fetch_data) # convert into numpy object

			yh = train[d * 18 + 9][start + PERIOD] # pm2.5 right answer => y hat
			print("y hat=", yh)
			
			dot_result = predict(np_data, w)
			print("predict=", dot_result)
			gradient_of_error = -2.0 * ( yh - dot_result) * np_data

			# compute new w
			w = w - ETA * gradient_of_error
			print("w =", w.round(2))

			Ein = ( yh - predict(np_data, w)) ** 2
			print("Ein =", Ein)

			this_Ein.append(Ein)

		average_this_Ein = sum(this_Ein)/(HOURS_OF_DAY - PERIOD - 1)
		print("this_Ein =", average_this_Ein)
		
		all_Ein.append(average_this_Ein)

print("average Ein = ", sum(all_Ein)/(DATA_SIZE * ITERATION) )
