import numpy as np
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000*1000


def create_lexicon(pos, neg):
	lexicon = []
	for fl in (pos, neg):
		with open(fl, 'r') as f:
			content = f.readlines()
			for ln in content[:hm_lines]:
				all_words = word_tokenize(ln.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)

	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	return l2


def sample_handling(sample, lexicon, classification):
	feature_set = []
	with open(sample, 'r') as f:
		content = f.readlines()
		for ln in content[:hm_lines]:
			current_words = word_tokenize(ln.lower())
			current_words = [lemmatizer.lemmatize(it) for it in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			feature_set.append([features, classification])
	return feature_set


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling(pos, lexicon, (1, 0))
	features += sample_handling(neg, lexicon, (0, 1))
	random.shuffle(features)

	features = np.array(features)
	test_size = int(test_size * len(features))
	train_x = list(features[:, 0][:-test_size])
	train_y = list(features[:, 1][:-test_size])

	test_x = list(features[:, 0][-test_size:])
	test_y = list(features[:, 1][-test_size:])

	return train_x, train_y, test_x, test_y


if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('./../data/text_pos.txt',
																	  './../data/text_neg.txt')

	print(test_x[:10])