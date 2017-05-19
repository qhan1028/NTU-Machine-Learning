# ML 2017 hw4 problem 2.
# Visualization of Word Vectors

import word2vec as wv
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sys import argv
from os.path import isfile
from sklearn.manifold import TSNE
from nltk.corpus import brown
from adjustText import adjust_text

np.set_printoptions(precision=4, suppress=True)
SIZE = 1000


def filter_words(word, case):
	
	if len(word) < 2:
		return False
	
	for c in word:
		if c in ['\'', '\"', ',', '.', ':', ';', '!', '?', '_', '“', '”', '’', '‘', '/']:
			return False
	
	print(" " + str(case), end="")
	if case in ['NN', 'JJ', 'NNP', 'NNS', None]:
		return True
	else:
		return False


def main():
	
	if '--download-nltk' in argv:
		nltk.download('punkt')
		nltk.download('maxent_treebank_pos_tagger')
		nltk.download('averaged_perceptron_tagger')
		nltk.download('brown')

	if not isfile('wordvec.bin') or '--train' in argv:
		print("\nwords to phrases...")
		wv.word2phrase('./HarryPotter/HarryPotter.txt', 'phrase', verbose=1)
		print("\nphrases to vectors...")
		wv.word2vec('phrase', 'wordvec.bin', size=50, verbose=1)
		print("")

	print("\nload model...")
	model = wv.load('wordvec.bin')
	print("model shape: " + repr(model.vectors.shape))
	
	X, Y = [], []
	if '--load-vector' in argv:
		if isfile('X.npy') and isfile('Y.npy'):
			X = np.load('X.npy')
			Y = np.load('Y.npy')
		else:
			print("can't load X.npy, Y.npy")
			return
	else:
		print("TSNE...")
		tsne = TSNE(n_components=2, learning_rate=10, random_state=0)
		vectors = tsne.fit_transform(X=model.vectors[:SIZE, :])
		X = vectors[:, 0]
		Y = vectors[:, 1]

	print("start plot...(using nltk.corpus.brown)")
	brown_tagged_sents = brown.tagged_sents(categories='news')
	unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
	words = unigram_tagger.tag(model.vocab[:SIZE])
	texts = []
	plt.figure(figsize=(12, 8))

	for x, y, word in zip(X, Y, words):
		print("word: (%s, %s)" % (word[0], word[1]), end="")
		
		if filter_words(word[0], word[1]):
			print("\r\t\t\t\tplot")
			plt.plot(x, y, 'o')
			texts.append(plt.text(x, y, word[0], fontsize=8))

		else:
			print("\r\t\t\t\tignore")
	
	adjust_text(texts, force_text=1, arrowprops=dict(arrowstyle="-", color="k", lw=1))

	plt.savefig("wordvec.png", dpi=100)
	plt.show()


if __name__ == '__main__':
	main()
