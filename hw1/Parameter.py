# ML 2017 hw1
# Parameter Parser

import csv
import numpy as np

def read_parameter(file_name):

	ITERATION = 0
	ETA = 0
	PERIOD = 0

	FEATURE = []
	NUM_FEATURE = 0

	VALIDATION = 0

	SGD = 0
	BATCH = 0

	w = []
	b = 0.0

	with open(file_name, "r", encoding="big5") as f:

		for line in list(csv.reader(f)):

			if line[0] == "iteration":
				ITERATION = int(line[1])

			elif line[0] == "eta":
				ETA = float(line[1])

			elif line[0] == "period":
				PERIOD = int(line[1])

			elif line[0] == "feature":
				FEATURE = [int(x) for x in line[1:] if x != ""]

			elif line[0] == "validation":
				VALIDATION = int(line[1])

			elif line[0] == "sgd":
				SGD = 1

			elif line[0] == "batch":
				BATCH = int(line[1])

			elif line[0] == "w":
				w.append([float(x) for x in line[1:] if x != ""])

			elif line[0] == "b":
				b = float(line[1])

			else:
				continue

	NUM_FEATURE = len(FEATURE)
	w = np.array(w)

	return ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, FEATURE, NUM_FEATURE, w, b
