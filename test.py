'''
INTERACTIVE
'''

import pickle
import pycrfsuite
from feats import sent2features, sent2labels, sent2tokens, pos_feats, pos_word_feats, crf_feat
from six import text_type
from os.path import join
from nltk import word_tokenize

MODEL_ROOT = './models/'

def load_model(model_name):
	model_path = join(MODEL_ROOT, model_name+'.pickle')
	print model_path
	with open(model_path) as f:
		classifier = pickle.load(f)
		return classifier


nb_classifier = load_model('slnb_1')
me_classifier = load_model('sl_maxent_1')

crf_classifer = pycrfsuite.Tagger()
crf_classifer.open(join(MODEL_ROOT, 'sl_1.crfsuite'))

while True:
	try:
		sentence = raw_input("> ")
		if not issubclass(type(sentence), text_type):
	  		sentence = text_type(sentence, encoding='ascii', errors='replace')
	except EOFError:
		break
	if not sentence:
		break
	# Do something to process command line input from user

	print "Naive Bayes ==> ", nb_classifier.classify(pos_word_feats(sentence))
	print "MaxEnt      ==> ", me_classifier.classify(pos_word_feats(sentence))
	print "CRF         ==> ", crf_classifer.tag(sent2features(crf_feat(sentence)))[-1]
