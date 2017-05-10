# ML hw4 problem 3.

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def elu(arr):
	return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
	w = np.random.normal(scale=0.5, size=(in_size, out_size))
	b = np.random.normal(scale=0.5, size=out_size)
	return (w, b)


def forward(inpd, layers):
	out = inpd
	for layer in layers:
		w, b = layer
		out = elu(out @ w + b)

	return out


def gen_data(dim, layer_dims, N):
	layers = []
	data = np.random.normal(size=(N, dim))

	nd = dim
	for d in layer_dims:
		layers.append(make_layer(nd, d))
		nd = d

	w, b = make_layer(nd, nd)
	gen_data = forward(data, layers)
	gen_data = gen_data @ w + b
	return gen_data


if __name__ == '__main__':

	if '--load-C' in argv:
		print("load centers...")
		centers = np.load('C.npy')
		centers = centers.reshape(-1,)
		plt.plot(centers, 'b')
		plt.title('100 iterations')
		plt.xlabel('dimension')
		plt.ylabel('average std')
		plt.savefig('centers.png')
		plt.show()

	print("generate centers in 1 iteration...")
	V_std = []
	for d in range(1, 61):
		print("\rdimension: %d" % d, end="", flush=True)
		N = np.random.randint(1e4, 1e5)
		layer_dims = [np.random.randint(60, 80), 100]
		data = gen_data(d, layer_dims, N)
		V_std.append( data.std() )
	print("")

	V_std_mean = []
	for d in range(1, 61):
		print("\rdimension: %d" % d, end="", flush=True)
		N = np.random.randint(1e4, 1e5)
		layer_dims = [np.random.randint(60, 80), 100]
		data = gen_data(d, layer_dims, N)
		V_std_mean.append( np.mean( data.std(axis=0) ) )

	plt.plot(V_std, 'b')
	plt.plot(V_std_mean, 'r')
	plt.legend(['all std', 'std mean'], loc='upper left')
	plt.xlabel('dimension')
	plt.ylabel('std')
	plt.savefig('gen.png')
	plt.show()
