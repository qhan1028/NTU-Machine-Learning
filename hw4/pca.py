# ML 2017 hw4 problem 1.
# Principle Component Analysis (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from PIL import Image
from numpy.linalg import svd, eig


def read_image(subjects, indices):
	
	image_dir = './faceExpressionDataBase/'
	data = []

	for s in subjects:

		for i in indices:
			filename = image_dir + s + i + '.bmp'
			im = Image.open(filename)
			p = np.array(im)
			data.append(p)

	return data


def main():
	
	subjects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	indices = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
	imgs = read_image(subjects, indices)

	# problem 1. find first 9 eigenfaces
	print("Problem 1.")
	imgs = np.array(imgs, np.float32).reshape((-1, 64 * 64))
	imgs_mean = imgs.mean(axis=0)
	U, s, V = svd(imgs - imgs_mean, full_matrices=False)
	eigenvectors = V
	S = np.diag(s)

	plt.imshow(imgs_mean.reshape(64, 64), cmap='gray')
	plt.title("Average Image")
	plt.savefig("pca_1_1.png")

	for i in range(9):
		print("\reigenface: %d" % (i+1), end="", flush=True)
		plt.subplot(3, 3, i+1)
		plt.title("%f" % s[i], fontsize=8)
		plt.imshow(eigenvectors[i, :].reshape(64, 64), cmap='gray')
		plt.xticks([])
		plt.yticks([])
	print("")
	plt.savefig("pca_1_2.png")

	# problem 2. project image onto first 5 eigenfaces and reconstruct
	print("Problem 2.")
	plt.figure(figsize=(10, 10))

	for i in range(100):
		print("\rplot original image: %d" % (i+1), end="", flush=True)
		plt.subplot(10, 10, i+1)
		img = imgs[i]
		plt.imshow(img.reshape(64, 64), cmap='gray')
		plt.title("%s%s" % (subjects[i // 10], indices[i % 10]))
		plt.xticks([])
		plt.yticks([])

	print("")
	plt.tight_layout(h_pad=0.5, w_pad=0.1)
	plt.savefig("pca_2_1.png")
		
	plt.figure(figsize=(10, 10))

	for i in range(100):
		print("\rreconstruct image: %d" % (i+1), end="", flush=True)
		plt.subplot(10, 10, i+1)
		img = imgs[i]
		project = np.dot(eigenvectors[:5, :], img - imgs_mean)
		reconstruct = np.dot(eigenvectors[:5, :].T, project) + imgs_mean
		RMSE = np.sqrt( np.sum((img - reconstruct) ** 2) / (64 * 64) ) / 255.
		
		plt.imshow(reconstruct.reshape(64, 64), cmap='gray')
		plt.title("RMSE = %.2f" % RMSE, fontsize=8)
		plt.xticks([])
		plt.yticks([])
	
	print("")
	plt.tight_layout(h_pad=0.5, w_pad=0.1)
	plt.savefig("pca_2_2.png")

	# problem 3. find top k eigenfaces s.t. all RMSE is less than 1%
	print("Problem 3.")
	top_k, RMSE_result = 1, 1
	for k in range( len(eigenvectors) ):
		
		RMSE = 0
		for i in range(100):
			print("\rk = %d, reconstruct image: %d" % (k+1, i+1), end="", flush=True)
			img = imgs[i]
			project = np.dot(eigenvectors[:k+1, :], img - imgs_mean)
			reconstruct = np.dot(eigenvectors[:k+1, :].T, project) + imgs_mean
			RMSE += np.sqrt( np.sum( (img - reconstruct) ** 2) / (64 * 64))

		RMSE /= (255 * 100)
		print(", RMSE = %.4f" % RMSE, end="", flush=True)
		if RMSE < 0.01:
			top_k = k+1
			print(", top k = %d" % top_k)
			break
	
	return

	print("Single k of each image")
	plt.figure(figsize=(10, 10))

	for i in range(100):
		plt.subplot(10, 10, i+1)
		img = imgs[i]

		RMSE = np.inf
		top_k = len(eigenvectors)
		R = img

		for k in range(len(eigenvectors)):
			print("\rreconstruct image: %d, k = %d" % (i+1, k+1), end="", flush=True)
			project = np.dot(eigenvectors[:k+1, :], img - imgs_mean)
			reconstruct = np.dot(eigenvectors[:k+1, :].T, project) + imgs_mean
			RMSE = np.sqrt( np.sum((img - reconstruct) ** 2) / (64 * 64) ) / 255.

			if RMSE < 0.01:
				top_k = k+1
				R = reconstruct
				break

		plt.imshow(R.reshape(64, 64), cmap='gray')
		plt.title("k = %d" % (top_k))
		plt.xticks([])
		plt.yticks([])

	print("")
	plt.tight_layout(h_pad=0.5, w_pad=0.1)
	plt.savefig("pca_3.png")


if __name__ == "__main__":
	main()
