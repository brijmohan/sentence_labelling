
from itertools import chain
import nltk
from nltk import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import random

from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold

print(sklearn.__version__)

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

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
	print "Class ==> {}, #samples ==> {}".format(k, len(v))
	# 20% of balanced data reserved for training, rest for testing
	train_idx = int(len(v) - len(v) / 5.0)
	train_set += [(sent, k) for sent in v[:train_idx]]
	test_set += [(sent, k) for sent in v[train_idx:]]


def pos_feats(sent):
	pos_arr = nltk.pos_tag(sent)
	feats = {}
	for pos in pos_arr:
		feats['pos({})'.format(pos[1].lower())] = True
	return feats


def word_feats(sent):
	feats = {}
	for word in nltk.word_tokenize(sent):
		feats['word({})'.format(word.lower())] = True
	return feats


def pos_word_feats(sent):
	text = word_tokenize(sent)
	pos_arr = nltk.pos_tag(text)
	feats = {}
	for word in pos_arr:
		feats['has({})'.format(word[0].lower())] = True
		feats['has({})'.format(word[1].lower())] = True
	return feats


def crf_feats(dataset):
	crf_ds = []
	for sent, label in dataset:
		#print sent
		sent_feats = []
		text = word_tokenize(sent)
		pos_arr = nltk.pos_tag(text)
		#print pos_arr
		for word in pos_arr:
			#print word
			sent_feats.append((word[0].lower(), word[1].lower(), label)) 
		crf_ds.append(sent_feats)

	return crf_ds


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    print sent
    return [token for token, postag, label in sent]


print "\nExtracting word features (a length(sentence) size dictionary containing True if word is present in sentence)."
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

print train_set[0]

train_sents = crf_feats(train_set)
test_sents = crf_feats(test_set)

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

#print y



#X = X_train+X_test
#y = y_train+y_test

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    print tagset
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


kf = KFold(n_splits=3)
ssf = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
skf = StratifiedKFold(n_splits=5)

print X[0]
#for kidx, (train, test) in enumerate(kf.split(X)):
for kidx, (train, test) in enumerate(skf.split(X, [l[0] for l in y])):

	print "============training for fold {} ...============".format(kidx)

	trainer = pycrfsuite.Trainer(verbose=False)

	nb_train_feats = []
	nb_test_feats = []
	#print train
	for tidx in train:
	    trainer.append(X[tidx], y[tidx])
	    nb_train_feats.append((total_featureset[tidx][0], total_featureset[tidx][1]))

	for tidx in test:
	    nb_test_feats.append((total_featureset[tidx][0], total_featureset[tidx][1]))

	print "\nTraining Naive Bayes Classifier."
	classifier = nltk.NaiveBayesClassifier.train(nb_train_feats)

	print "Testing Naive Bayes."
	print("Accuracy: {}".format(nltk.classify.accuracy(classifier, nb_test_feats)))


	print "Training Maxent Classifier."
	classifier = nltk.classify.MaxentClassifier.train(nb_train_feats, 'GIS', trace=0, max_iter=500)
	print "Testing Maxent Classifier."
	print("Accuracy: {}".format(nltk.classify.accuracy(classifier, nb_test_feats)))


	trainer.set_params({
	    'c1': 1.0,   # coefficient for L1 penalty
	    'c2': 1e-3,  # coefficient for L2 penalty
	    'max_iterations': 50,  # stop earlier

	    # include transitions that are possible, but not observed
	    'feature.possible_transitions': True
	})


	print trainer.params()

	trainer.train('sentence_label_{}.crfsuite'.format(kidx))

	#print trainer.logparser.last_iteration

	#print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]

	
	tagger = pycrfsuite.Tagger()
	tagger.open('sentence_label_{}.crfsuite'.format(kidx))


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

	print(bio_classification_report(y_test, y_pred))


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
