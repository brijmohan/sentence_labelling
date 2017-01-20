import itertools
import nltk
from nltk import word_tokenize
import sklearn
from sklearn import datasets
import pycrfsuite
import random

import numpy as np

from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from six.moves import xrange, input  # pylint: disable=redefined-builtin
from six import text_type
from os.path import join

from utils import plot_confusion_matrix, label_classification_report, print_cm
from corpus import load_corpus
from feats import sent2features, sent2labels, sent2tokens, pos_feats, pos_word_feats, crf_feats
import pickle

print(sklearn.__version__)

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


label_set, train_set, test_set = load_corpus()

MODEL_ROOT='./models/'

print "\nExtracting word features ..."

# uncomment below lines to use these features but they perform inferior to the ones used in demo
# WORD FEATS
#train_featuresets = [(word_feats(sample[0]), sample[1]) for sample in train_set]
#test_featuresets = [(word_feats(sample[0]), sample[1]) for sample in test_set]

# POS FEATS
#train_featuresets = [(pos_feats(sample[0]), sample[1]) for sample in train_set]
#test_featuresets = [(pos_feats(sample[0]), sample[1]) for sample in test_set]

# WORD+POS FEATS
train_featuresets = [(pos_word_feats(sample[0]), sample[1]) for sample in train_set]
test_featuresets = [(pos_word_feats(sample[0]), sample[1]) for sample in test_set]

total_featureset = train_featuresets+test_featuresets


#print train_set[0]

train_sents = crf_feats(train_set)
test_sents = crf_feats(test_set)

print "\n\nHere's how the feature set looks like...\n\n"
print train_sents[0]

total_sents = train_sents+test_sents

'''
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
'''

X = [sent2features(s) for s in total_sents]
y = [sent2labels(s) for s in total_sents]

def save_model(filename, classifier):
	with open(join(MODEL_ROOT, filename+'.pickle'), 'wb') as f:
		pickle.dump(classifier, f)

'''
Train Naive Bayes and MaxEnt Classifier
'''
def train_nb_maxent(fold, nb_train_feats, nb_test_feats, y_test):

	print "\nTraining Naive Bayes Classifier."
	print "=================================="
	classifier = nltk.NaiveBayesClassifier.train(nb_train_feats)

	save_model('slnb_{}'.format(fold), classifier)

	print "Testing Naive Bayes."
	print "=================================="
	nbacc = nltk.classify.accuracy(classifier, nb_test_feats)
	print("Accuracy: {}".format(nbacc))

	y_pred = [classifier.classify(sent_feat[0]) for sent_feat in nb_test_feats]
	print_cm(y_test, y_pred)

	# Plot normalized confusion matrix
	#plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=label_set, normalize=True,
	                      #title='Normalized confusion matrix')

	#plt.show()

	print "Training Maxent Classifier. Press Ctrl+C to stop training and see model predition."
	print "=================================================================================="
	classifier = nltk.classify.MaxentClassifier.train(nb_train_feats, 'GIS', trace=0, max_iter=500)
	save_model('sl_maxent_{}'.format(fold), classifier)
	print "Testing Maxent Classifier."
	print "=========================="
	meacc = nltk.classify.accuracy(classifier, nb_test_feats)
	print("Accuracy: {}".format(meacc))

	y_pred = [classifier.classify(sent_feat[0]) for sent_feat in nb_test_feats]
	print_cm(y_test, y_pred)

	return nbacc, meacc


def train_crf(trainer):
	trainer.set_params({
	    'c1': 1.0,   # coefficient for L1 penalty
	    'c2': 1e-3,  # coefficient for L2 penalty
	    'max_iterations': 50,  # stop earlier

	    # include transitions that are possible, but not observed
	    'feature.possible_transitions': True
	})


	print "CRF Training Parameters"
	print "======================="
	print trainer.params()

	trainer.train(join(MODEL_ROOT, 'sl_{}.crfsuite'.format(kidx)))

	#print trainer.logparser.last_iteration

	#print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]

	
	tagger = pycrfsuite.Tagger()
	tagger.open(join(MODEL_ROOT, 'sl_{}.crfsuite'.format(kidx)))


	#example_sent = test_sents[random.randint(0, len(test_sents))]
	example_sent = total_sents[random.choice(test)]
	print(' '.join(sent2tokens(example_sent)) + '\n\n')

	print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
	print("Correct:  ", ' '.join(sent2labels(example_sent)))


	y_pred = []
	y_test = []
	for teidx in test:
		y_pred.append(tagger.tag(X[teidx]))
		y_test.append(y[teidx])

	acc, cr = label_classification_report(y_test, y_pred)
	print cr


	'''
	from collections import Counter
	info = tagger.info()

	def print_transitions(trans_features):
	    for (label_from, label_to), weight in trans_features:
	        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

	print("Top likely transitions:")
	print_transitions(Counter(info.transitions).most_common(15))

	print("\nTop unlikely transitions:")
	print_transitions(Counter(info.transitions).most_common()[-15:])



	def print_state_features(state_features):
	    for (attr, label), weight in state_features:
	        print("%0.6f %-6s %s" % (weight, label, attr))    

	print("Top positive:")
	print_state_features(Counter(info.state_features).most_common(20))

	print("\nTop negative:")
	print_state_features(Counter(info.state_features).most_common()[-20:])
	'''

	return acc


#print y


#X = X_train+X_test
#y = y_train+y_test


#kf = KFold(n_splits=3)
#ssf = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
print "Splitting the data set in 5 folds created by preserving the \%age of samples in each class"
skf = StratifiedKFold(n_splits=5)

best_nb = 0
best_me = 0
best_crf = 0

#print X[0]
#for kidx, (train, test) in enumerate(kf.split(X)):
for kidx, (train, test) in enumerate(skf.split(X, [l[0] for l in y])):

	print "============training for fold {} ...============".format(kidx)

	trainer = pycrfsuite.Trainer(verbose=False)
	nb_train_feats = []
	nb_test_feats = []
	y_test = []
	#print train
	for tidx in train:
	    trainer.append(X[tidx], y[tidx])
	    nb_train_feats.append((total_featureset[tidx][0], total_featureset[tidx][1]))

	for tidx in test:
	    nb_test_feats.append((total_featureset[tidx][0], total_featureset[tidx][1]))
	    y_test.append(total_featureset[tidx][1])


	nbacc, meacc = train_nb_maxent(kidx, nb_train_feats, nb_test_feats, y_test)
	crfacc = train_crf(trainer)

	best_nb = kidx if nbacc > best_nb else best_nb
	best_me = kidx if meacc > best_me else best_me
	best_crf = kidx if crfacc > best_crf else best_crf


with open(join(MODEL_ROOT, 'best.txt'), 'wb') as bm:
	bm.write('NaiveBayes\tslnb_{}\n'.format(best_nb))
	bm.write('MaxEnt\tsl_maxent_{}\n'.format(best_nb))
	bm.write('CRF\tsl_{}.crfsuite\n'.format(best_crf))
