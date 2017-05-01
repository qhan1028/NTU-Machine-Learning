# ML2017 hw3 Plot Model

from sys import argv
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model
from keras.utils import plot_model, np_utils
from sklearn.metrics import confusion_matrix


READ_FROM_NPZ = 1
CATEGORY = 7
SHAPE = 48

STRUCTURE = 0
CONFUSION = 0
SALIENCY = 0

def read_train(filename):
	
	X, Y = [], []
	with open(filename, "r", encoding="big5") as f:
		count = 0
		for line in list(csv.reader(f))[1:]:
			Y.append( float(line[0]) )
			X.append( [float(x) for x in line[1].split()] )
			count += 1
			print("\rX_train: " + repr(count), end="", flush=True)
		print("", flush=True)

	return np.array(X), np_utils.to_categorical(Y, CATEGORY)

# argv: [1]train.csv [2]model.h5
def main():

	X, Y = [], []
	if READ_FROM_NPZ:
		print("read from npz...")
		data = np.load("data.npz")
		X = data['arr_0']
		Y = data['arr_1']
	else:
		print("read train data...")
		X, Y = read_train(argv[1])

	X = X / 255
	X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)
	
	print("load model...")
	model_name = argv[2]
	model = load_model(model_name)
	model.summary()

	if STRUCTURE:
		print("plot structure...")
		#plot_model(model, show_layer_names=False, show_shapes=True, to_file=model_name[:-3] + "_struct.png")

	if CONFUSION:
		print("plot confusion matrix...")
		Y_1D = np.argmax(Y, 1)
		print("predict train data...")
		predict = model.predict(X, verbose=1, batch_size=32)
		Y_predict = np.argmax(predict, 1)
		cm = confusion_matrix(Y_1D, Y_predict)
		print(cm)

		sb.set(font_scale=1.4)
		figure = sb.heatmap(cm, annot=True, annot_kws={"size": 16 }, fmt='d', cmap='YlGnBu')
		figure.set_xticklabels(["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"])
		figure.set_yticklabels(["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"])
		plt.yticks(rotation=0)
		plt.xlabel("Predicted Label")
		plt.ylabel("True Label")
		figure.get_figure().savefig(model_name[:-3] + "_cm.png")

	if SALIENCY:
		fig_no = [(1, 921), (2, 1108), (3, 33), (4, 119), (5, 1028)]
		fig_num = len(fig_no)
		plt.clf()
		for i, f in fig_no:
			plt.subplot(2, fig_num, i)
			plt.title("Figure " + repr(f) + ".")
			plt.imshow(X[f].reshape(SHAPE, SHAPE), cmap='gray')
			ax = plt.gca()
			ax.set_xticks(range(1, 48, 10))
			ax.set_yticks(range(1, 48, 10))
			ax.set_xticklabels(range(1, 48, 10))
			ax.set_yticklabels(range(1, 48, 10))

			print("plot saliency maps on image " + repr(f) + "...")
			plt.subplot(2, fig_num, i+fig_num)
			X_fig = X[f].reshape(1, SHAPE, SHAPE, 1)
			Y_fig = Y[f].reshape(1, CATEGORY)
			score = model.evaluate(X_fig, Y_fig, verbose=1)
			original_loss = score[0]
			loss_matrix = np.zeros([SHAPE, SHAPE])
			for i in range(SHAPE):
				for j in range(SHAPE):
					print("\rrow: " + repr(i) + " column: " + repr(j), end="", flush=True)
					X_tmp = np.array(X_fig)
					X_tmp[0][i][j][0] += 1
					score = model.evaluate(X_tmp, Y_fig, verbose=0)
					loss_matrix[i][j] = (score[0] - original_loss) ** 2
			print("", flush=True)

			plt.imshow(loss_matrix, cmap='cool', vmin=loss_matrix.min(), vmax=loss_matrix.max())
			ax = plt.gca()
			ax.set_xticks(range(1, 48, 10))
			ax.set_yticks(range(1, 48, 10))
			ax.set_xticklabels(range(1, 48, 10))
			ax.set_yticklabels(range(1, 48, 10))
		plt.savefig(model_name[:-3] + "_sm.png")

if __name__ == "__main__":
	main()
