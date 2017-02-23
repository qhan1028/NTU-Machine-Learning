import numpy as np
import sys

x = []
with open(sys.argv[1], 'r') as f:
	for line in f:
		x.append([int(i) for i in line.split(',')])

y = []
with open(sys.argv[2], 'r') as f:
	for line in f:
		y.append([int(i) for i in line.split(',')])

x = np.array(x)
y = np.array(y)

result = np.dot(x, y)
result = np.reshape(result, np.size(result))
result = np.sort(result)
f = open('ans_one.txt', 'w')
for n in result:
	print(n, file=f)
