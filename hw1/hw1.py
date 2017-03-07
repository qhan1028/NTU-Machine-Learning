# ML 2017 hw1 

import numpy as np
import sys
import csv

data = []
with open(sys.argv[1], 'rb') as f:
	data = f.read().splitlines()
	for line in data:
		l = str(line).split(',')
		l = [i.replace("\"", "") for i in l]
		print(l)


