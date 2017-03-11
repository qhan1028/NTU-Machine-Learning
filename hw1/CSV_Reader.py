# ML 2017 hw1
# CSV reader without importing csv

# read from csv and trim all chinese words
def read_data(train_file, test_file):

	train = [[] for _ in range(18)]
	with open(train_file, 'rb') as f:
		data = f.read().splitlines()
		i = 0
		for line in data[1:]: # trim the first data which is header
			line = [x.replace("\'", "") for x in str(line).split(',')[3:]]

			if i % 18 == 10:
				line = [x.replace("NR", "0") for x in line]

			line = [float(x) for x in line]
			train[i % 18] += line
			
			i += 1

	test = []
	with open(test_file, 'rb') as f:
		data = f.read().splitlines()
		i = 0
		for line in data: # trim the first data which is header
			line = [x.replace("\'", "") for x in str(line).split(',')[2:]]

			if i % 18 == 10:
				line = [x.replace("NR", "0") for x in line]

			line = [float(x) for x in line]
			test.append(line)

			i += 1

	return train, test

from sys import argv
import csv

def read_data2():

	NUM_FEATURE = 18
	data = [ [] for _ in range(NUM_FEATURE) ]

	# need big-5 encoding because of chinese
	with open(argv[1], "r", encoding="big5") as f:

		i = 0
		for line in list(csv.reader(f)):

			# there are "NR" in some rows
			data[i % 18] += [float(x.replace("NR", "0")) for x in line[3:]]
			i += 1

	print(data)
