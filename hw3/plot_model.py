# ML2017 hw3 Plot Model

from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model
from keras.utils import plot_model

# argv: [1]model.h5
def main():
	
	filename = argv[1]
	print("load model...")
	model = load_model(filename)
	model.summary()
	print("save fig...")
	plot_model(model, show_layer_names=False, show_shapes=True, to_file=filename[:-3] + ".png")

if __name__ == "__main__":
	main()
