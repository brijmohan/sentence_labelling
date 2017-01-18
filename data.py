import nltk

datafile = 'LabelledData.txt'

labels = {}

print "Reading complete dataset."
with open(datafile) as dfile:
	for line in dfile.read().splitlines():
		line = line.strip()
		if line:
			sp = line.split(',,,')
			l = sp[1].strip()
			if l not in labels:
				labels[l] = []
			labels[l].append(sp[0].strip())

#print labels
label_set = set(labels.keys())

print "Splitting into train (80%) and test (20%)."
train_set = []
test_set = []
for k, v in labels.items():
	# 20% of balanced data reserved for training, rest for testing
	train_idx = int(len(v) - len(v) / 5.0)
	train_set += [(sent, k) for sent in v[:train_idx]]
	test_set += [(sent, k) for sent in v[train_idx:]]

def word_feats(sent):
	feats = {}
	for word in nltk.word_tokenize(sent):
		feats['has({})'.format(word.lower())] = True
	return feats

print "\nExtracting word features (a vocabulary size dictionary containing True if word is present in sentence)."
train_featuresets = [(word_feats(sample[0]), sample[1]) for sample in train_set]
test_featuresets = [(word_feats(sample[0]), sample[1]) for sample in test_set]

print "\nTraining Naive Bayes Classifier."
classifier = nltk.NaiveBayesClassifier.train(train_featuresets)

print "Testing Naive Bayes."
print("Accuracy: {}".format(nltk.classify.accuracy(classifier, test_featuresets)))


print "Training Maxent Classifier."
classifier = nltk.classify.MaxentClassifier.train(train_featuresets, 'GIS', trace=0, max_iter=1000)
print "Testing Maxent Classifier."
print("Accuracy: {}".format(nltk.classify.accuracy(classifier, test_featuresets)))