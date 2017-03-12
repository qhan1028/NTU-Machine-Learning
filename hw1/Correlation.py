# ML 2017 hw1
# Calculating Correlation Coefficient

import numpy as np
from sys import argv
from CSV_Reader import *

np.set_printoptions(precision = 4, suppress = True)

train, test = read_data(argv[1], argv[2])

cor = np.corrcoef(train[9], [train[i] for i in range(18)])
print("Correlation Coefficient with PM2.5:\n")
for i in range(1, len(cor[0])):
	print("feature " + repr(i - 1) + " : " + repr(cor[0][i]))
