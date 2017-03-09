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
		if i < 18:
			train.append(line)
		else:
			train[i % 18] += line
		
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

np.set_printoptions(precision = 2, suppress = True)

# define constants
ITERATION = 1000
ETA = 0.000000005
VALIDATION = 500

MAX_TIME = int(len(train[0]))
PERIOD = 9

SF = [7, 8, 9, 16] # selected feature
#SF = range(18) # selected feature
NUM_FEATURE = len(SF)
w = [ [0.3] * PERIOD ] * NUM_FEATURE # feature * (constant + period)
b = 3.0#np.random.random()
#w = np.random.random(np.shape(w))
w = np.array(w)

print("iteration =", ITERATION)
print("eta =", ETA)
print("validation =", VALIDATION)
print("selected features =", SF)
print("w =", w)
print("b =", b)

def filter_data(SF, d):
	
	result = []
	for sf in SF:
		
		result.append(train[d * 18 + sf])
	
	return result

def filter_hours(data, start, period, selected_features):

	result = []
	for f in selected_features:
		result += [data[f][start : start + period]]
	
	return result

def predict(X, w, b):
	
	Y = np.sum(X * w) + b
	return Y

# train
all_Ein = []
for i in range(ITERATION):

	if i % 100 == 0:
		print("progress:", i)
	
	iter_Ein = []
	sum_gradient_X = np.zeros([NUM_FEATURE, PERIOD])
	sum_gradient_b = 0.0

	for start in range(MAX_TIME - PERIOD - 1)[VALIDATION:]:

		X = np.array( filter_hours(train, start, PERIOD, SF) )
		yh = train[9][start + PERIOD]

		sum_gradient_X += (-2.) * (yh - predict(X, w, b)) * X
		sum_gradient_b += (-2.) * (yh - predict(X, w, b))

		Etrain = (yh - predict(X, w, b)) ** 2
		iter_Ein.append(Etrain)

	current_Ein = np.mean(iter_Ein)
	all_Ein.append(current_Ein)

	# update parameters
	w = w - ETA * sum_gradient_X
	b = b - ETA * sum_gradient_b
	
	if i % 10 == 0:
		print("current Ein =", np.sqrt(current_Ein))

# print result
print("iteration =", ITERATION)
print("eta =", ETA)
print("validation num =", VALIDATION)
print("selected features =", SF)
print("average Ein = ", np.sqrt(np.mean(all_Ein)))
print("w = \n", w)
print("b =", b.round(4))

# validation
Evalid = []
for start in range(MAX_TIME - PERIOD - 1)[-VALIDATION:]:

	X = np.array( filter_hours(train, start, PERIOD, SF) )
	yh = train[9][start + PERIOD]

	Ev = (yh - predict(X, w, b)) ** 2
	Evalid.append(Ev)

print("Evalid =", np.sqrt(np.mean(Evalid)))


f = open(sys.argv[3], "w")
f.write("id,value\n")
# test
for d in range(240):
	
	test_data = filter_hours(test[ d*18 : d*18 + 18], 0, PERIOD, SF)
	np_data = np.array(test_data)

	dot_result = int(predict(np_data, w, b))
	if dot_result < 0:
		dot_result = -1

	f.write("id_" + repr(d) + "," + repr(dot_result) + "\n")
