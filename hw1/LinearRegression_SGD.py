# ML 2017 hw1 
# Linear Regression with Stochastic Gradient Descent (SGD)
# Validation (Ev)

from sys import argv
from random import shuffle
import numpy as np

from Parameter import read_parameter
from CSV_Reader import read_data 

# train and test data
train = [ [] for _ in range(18)] # FEATURE * MAX_TIME matrix
test = []

# default parameters
ITERATION = 4500
ETA = 1.25e-8
VALIDATION = 0 # validation size
DATA_SIZE = 1

SGD = 0
BATCH = 500

PERIOD = 7
MAX_TIME = 0

#FEATURE = range(18)
FEATURE = [7, 9, 12] # selected feature
NUM_FEATURE = len(FEATURE)

# default starting point
w = np.array([[0.01] * PERIOD] * NUM_FEATURE) # training maxtrix
b = 1.0 # bias

# set numpy print option
np.set_printoptions(precision = 6, suppress = True)


def print_message():
	print("\nLinear Regression: Stochastic Gradient Descent" if SGD else "\nLinear Regression")
	print("iteration =", ITERATION)
	print("eta =", ETA)
	print("max time=", MAX_TIME);
	print("validation =", VALIDATION)
	print("selected features =", FEATURE)
	print("period =", PERIOD)
	print("w =\n", w)
	print("b =", b)

def filter_hours(data, start, period, selected_features):
	result = []
	for f in selected_features:
		result += [data[f][start : start + period]]
	
	return result

def predict(X, w, b):
	Y = np.sum(X * w) + b
	return Y

def train_LR():
	global train, test, w, b

	all_Ein = []
	for i in range(ITERATION):

		if i % 100 == 0:
			print("progress:", i)
		
		sum_gradient_X = np.zeros([NUM_FEATURE, PERIOD]) 
		sum_gradient_b = 0.0
		iter_Ein = []
		for start in range(MAX_TIME - PERIOD - 1)[VALIDATION:]:

			X = np.array( filter_hours(train, start, PERIOD, FEATURE))
			yh = train[9][start + PERIOD]

			sum_gradient_X += (-2.) * (yh - predict(X, w, b)) * X
			sum_gradient_b += (-2.) * (yh - predict(X, w, b))

			Ein = (yh - predict(X, w, b)) ** 2
			iter_Ein.append(Ein)

		# update parameters
		w = w - ETA * sum_gradient_X
		b = b - ETA * sum_gradient_b

		all_Ein.append(iter_Ein)
		
		if i % 10 == 0:
			print("current Ein =", np.sqrt(np.mean(all_Ein[-10:])))
	
	return all_Ein

def train_SGD():
	global train, test, w, b
	
	index = list(range(MAX_TIME - PERIOD - 1))[VALIDATION:]
	size_index = len(index)
	shuffle(index) # random index if using SGD
	
	all_Ein = []
	for i in range(ITERATION):

		if i % 1000 == 0:
			print("progress:", i)
		
		sum_gradient_X = np.zeros([NUM_FEATURE, PERIOD])
		sum_gradient_b = 0.0
		batch_Ein = []
		for j in range(i * BATCH, (i+1) * BATCH):
			start = index[j % size_index]

			X = np.array( filter_hours(train, start, PERIOD, FEATURE))
			yh = train[9][start + PERIOD]

			sum_gradient_X += (-2.) * (yh - predict(X, w, b)) * X
			sum_gradient_b += (-2.) * (yh - predict(X, w, b))

			Ein = (yh - predict(X, w, b)) ** 2
			batch_Ein.append(Ein)

		all_Ein.append(batch_Ein)
		
		# update parameters every 50 data and reset sum
		w = w - ETA * sum_gradient_X
		b = b - ETA * sum_gradient_b
		
		if i % 100 == 0:
			print("current Ein =", np.sqrt(np.mean(all_Ein[-100:])))
		
	return all_Ein

def main():	
	global train, test, MAX_TIME
	train, test = read_data(argv[1], argv[2])
	MAX_TIME = int(len(train[0]) * (DATA_SIZE))

	global ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, FEATURE, NUM_FEATURE, w, b
	if len(argv) > 4: # if there is parameter file, load it
		ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, FEATURE, NUM_FEATURE, w, b = read_parameter(argv[4])
		predict_test()
		return
	
	print_message()
	all_Ein = train_SGD() if SGD else train_LR()
	print_message()

	average_Ein = np.sqrt(np.mean(all_Ein))
	final_Ein = np.sqrt(np.mean(all_Ein[-1000:])) if SGD else np.sqrt(np.mean(all_Ein[-10:]))
	Ev = validation()
	print("average Ein =", average_Ein, "\nfinal Ein =", final_Ein, "\nEvalid =", Ev)

	predict_test()
	#output_parameter(average_Ein, final_Ein, Ev)

def validation():
	Evalid = []
	for start in range(MAX_TIME - PERIOD - 1)[:VALIDATION]:

		X = np.array( filter_hours(train, start, PERIOD, FEATURE) )
		yh = train[9][start + PERIOD]

		Ev = (yh - predict(X, w, b)) ** 2
		Evalid.append(Ev)

	return np.sqrt(np.mean(Evalid)) if VALIDATION else 0

# output testing result (hw)
def predict_test():
	with open(argv[3], "w") as f:

		f.write("id,value\n")
		for d in range(240):

			data = np.array( filter_hours( test[ d * 18 : (d + 1) * 18], 9 - PERIOD, PERIOD, FEATURE))
			dot_result = int(predict(data, w, b))
			f.write("id_" + repr(d) + "," + repr(dot_result) + "\n")

def output_parameter(average_Ein, final_Ein, Ev):	
	filename = "SGD" if SGD else "LR"
	filename += "_" + repr(ITERATION) + "i"
	filename += "_" + repr(ETA)
	filename += "_" + repr(NUM_FEATURE) + "f"
	filename += "_" + "{:.3f}".format(final_Ein) + "Ein"

	with open("../hw1_test/params/" + filename + ".csv", "w") as f:
		
		f.write("iteration," + repr(ITERATION) + "\n")
		f.write("eta," + repr(ETA) + "\n")
		f.write("period," + repr(PERIOD) + "\n")
		f.write("sgd," + repr(SGD) + "\n")
		f.write("batch," + repr(BATCH) + "\n")
		
		f.write("feature,")
		for r in FEATURE:
			f.write(repr(r) + ",")
		f.write("\n")
		
		f.write("average Ein," + repr(average_Ein) + "\n")
		f.write("final Ein," + repr(final_Ein) + "\n")
		f.write("validation," + repr(VALIDATION) + "\n")
		f.write("Ev," + repr(Ev) + "\n")
		f.write("b," + repr(b) + "\n")

		for row in w:
			f.write("w,")
			for r in row:
				f.write(repr(r) + ",")
			f.write("\n")

if __name__ == "__main__":
	main()
