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

		if i % 18 == 10:
			line = [x.replace("NR", "0") for x in line]

		line = [float(x) for x in line]
		train.append(line)
		i += 1

test = []
with open(sys.argv[2], 'rb') as f:
	data = f.read().splitlines()
	i = 0
	for line in data: # trim the first data which is header
		line = [x.replace("\'", "") for x in str(line).split(',')[2:]]

		if i % 18 == 10:
			line = [x.replace("NR", "0") for x in line]

		line = [float(x) for x in line]
		test.append(line)
		i += 1


# define constants
ITERATION = 100
TRAIN_SIZE = int(len(train) / 18)
TEST_SIZE = int(len(test) / 18)
ETA = 0.0001

HOURS_OF_DAY = 24

#SF = [9] # selected feature
SF = range(18) # selected feature
NUM_FEATURE = len(SF)
PERIOD = 9
w = [ [1.0] * PERIOD + [1.0] ] * NUM_FEATURE # 9 hr pm2.5 feature and constant
w = np.array(w)
size_w = float(np.size(w))

print("iteration =", ITERATION)
print("eta =", ETA)
print("selected features =", SF)

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

# train
for i in range(ITERATION):

	if i % 100 == 0:
		print(i/100)
		if i % 1000 == 0:
			print(w.round(2))
	
	iter_Ein = []

	sum_gradient = np.zeros([NUM_FEATURE, PERIOD + 1])
	for d in range(TRAIN_SIZE):

		data = filter_data(SF, d)

		this_Ein = []
		for start in range(HOURS_OF_DAY - PERIOD - 1):
			
			fetch_data = filter_hours(data, start) # fetch hrs of data + constant
			np_data = np.array(fetch_data) # convert into numpy object

			yh = train[d * 18 + 9][start + PERIOD] # pm2.5 right answer => y hat
			# print("y hat=", yh)
			
			#sum_gradient += (-2.) * (yh - np.sum(np_data * w)/size_w ) * np_data
			gradient = (-2.) * (yh - predict(np_data, w)) * np_data
			w = w - ETA * gradient

			Ein = ( yh - predict(np_data, w)) ** 2
			# print("Ein =", Ein)

			this_Ein.append(Ein)

		average_this_Ein = sum(this_Ein)/(HOURS_OF_DAY - PERIOD - 1)
		# print("this_Ein =", average_this_Ein)	
		all_Ein.append(average_this_Ein)
		iter_Ein.append(average_this_Ein)

	# compute new w
	# gradient = sum_gradient / float(TRAIN_SIZE * (HOURS_OF_DAY - PERIOD - 1))
	# w = w - ETA * gradient
	# print("w =", w.round(2))
	if i % 10 == 0:
		print("current Ein = ", np.mean(iter_Ein))

print("final average Ein = ", np.mean(all_Ein))

print("final w = \n", w.round(2))

f = open(sys.argv[3], "w")

f.write("id,value\n")
# test
for d in range(TEST_SIZE):
	
	data = filter_data(SF, d)

	fetch_data = filter_hours(data, 9 - PERIOD)
	np_data = np.array(fetch_data)

	dot_result = int(predict(np_data, w))

	f.write("id_" + repr(d) + "," + repr(dot_result) + "\n")
