# ML 2017 hw4 problem 3.2
# Estimation of Intrinsic Dimension (Hand Rotation)

import math
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans


def read_images():
	
	image_dir = './hand/hand.seq'
	data = []

	for i in range(1, 482):
		print("\rread image...%d" % i, end="", flush=True)
		img = Image.open(image_dir + repr(i) + '.png')
		img_arr = np.array(img)
		data.append(img_arr)
	print("")

	return np.array(data)


def main():
	
	print("load hand image...")
	data = read_images()

	V = []
	for i in range(481):
		print("\rcompute variance...%d" % (i+1), end="", flush=True)
		S = data[i]
		V.append( S.std() )
	print("")
	V = np.array(V)

	print("sort variance...")
	sorted_index = np.argsort(V)
	unsorted_index = np.argsort(sorted_index)

	V = V.reshape(-1, 1)

	print("kmeans clustering with sorted variance...") 
	kmeans = KMeans(n_clusters=60, random_state=0).fit(np.sort(V, axis=0))
	dim = np.array(kmeans.labels_)

	print("rename dimension label...")
	renamed_dim = []
	prev, new_d = dim[0], 0
	for d in dim:
		if d == prev:
			renamed_dim.append(new_d)
		else:
			prev = d
			new_d += 1
			renamed_dim.append(new_d)

	print("unsort renamed dimension label...")
	result = np.array(renamed_dim)[unsorted_index]

	result_list = list(result)
	count = []
	for i in range(60):
		count.append(result_list.count(i))

	print('mode dimension = %d' % (np.argmax(np.array(count)) + 1) )
	plt.hist(result_list, bins=np.linspace(0, 60, 61))
	plt.xlabel('dimension')
	plt.ylabel('count')
	plt.savefig('hand_dimension.png')
	plt.show()


if __name__ == '__main__':
	main()

