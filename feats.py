import nltk
from nltk import word_tokenize

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
