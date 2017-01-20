datafile = './data/LabelledData.txt'

def load_corpus():
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

	return label_set, train_set, test_set