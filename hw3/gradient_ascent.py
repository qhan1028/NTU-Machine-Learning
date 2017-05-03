# ML 2017 hw3 gradient ascent

import numpy as np
from keras import backend as K
from keras.models import load_model
from scipy.misc import imsave
from sys import argv
import matplotlib.pyplot as plt

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def main():

	model_name = argv[1]
	model = load_model(model_name)
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	input_img = model.input

	layer_name = "conv2d_1"
	print("process on layer " + layer_name)
	filter_index = range(32)

	# for loop
	random_img = np.random.random((1, 48, 48, 1))
	for f in filter_index:
		print("process on filter " + repr(f))
		layer_output = layer_dict[layer_name].output

		loss = K.mean(layer_output[:, :, :, f])
		grads = K.gradients(loss, input_img)[0]
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		iterate = K.function([input_img], [loss, grads])

		input_img_data = np.array(random_img)

		step = 1.
		for i in range(20):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step
			print("\riteration: " + repr(i) + ", current loss: " + repr(loss_value), end="", flush=True)
			if loss_value <= 0:
				break
		print("", flush=True)

		img = input_img_data[0].reshape(48, 48)
		img = deprocess_image(img)
		plt.subplot(4, 8, f+1)
		plt.title(repr(f))
		plt.xticks([], [])
		plt.yticks([], [])
		plt.imshow(input_img_data[0].reshape(48, 48), cmap='gray')

	print("save image...")
	plt.savefig("%s_%s.png" % (model_name[:-3], layer_name))

if __name__ == "__main__":
	main()
