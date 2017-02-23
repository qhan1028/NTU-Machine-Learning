import numpy as np

x = []
with open('matrixA.txt', 'r') as f:
	for line in f:
		x = [int(i) for i in line.split(',')]

y = []
with open('matrixB.txt', 'r') as f:
	for line in f:
		y.append([int(i) for i in line.split(',')])

x = np.array(x)
y = np.array(y)

result = np.dot(x, y)
for n in result:
	print(n)
