# Comparative results of Naive Bayes, MaxEnt and Conditional Random Field (CRF) classifier for sentence labeling task 

This code trains probabilistic models to predict sentence labels amongst 5 categories. Namely, 'what', 'when', 'who', 'affirmation' and 'unknown'.

## Setup

Tested on a system with: Ubuntu 16.04, Python2, pip, virtualenv

```
sudo apt install python-pip virtualenv
```

Install this repo using following commands

```
git clone https://github.com/brijmohan/sentence_labelling.git
cd sentence_labelling/
bash install.sh
```

Activate virtual environment using:
```
. venv/bin/activate
```

## Testing trained models

To test the performance of trained models, launch testing console using the following command (Make sure virtual environment is activated). 
```
python test.py
```
Start typing your sentence next to '>' symbol and press Enter when finished. Testing models can be switched with other models. Best performing models are listed in ```models/best.txt``` file

Press Ctrl+D to exit the program.

## Training models over data

To launch training of models, enter following command (Make sure virtual environment is activated):
```
python train.py
```
Here's a brief explanation of the training output.

```
Reading complete dataset.
Splitting into train (80%) and test (20%).
Class ==> unknown, #samples ==> 272
Class ==> what, #samples ==> 609
Class ==> who, #samples ==> 402
Class ==> when, #samples ==> 96
Class ==> affirmation, #samples ==> 104

```
Shows the class distribution of data.

Sentence and its labels are transformed into tuples of word, its Part-of-Speech (POS) tag and true label.
```
('how did serfdom develop in and then leave russia ?', 'unknown')

[('how', 'wrb', 'unknown'), ('did', 'vbd', 'unknown'), ('serfdom', 'vb', 'unknown'), ('develop', 'vb', 'unknown'), ('in', 'in', 'unknown'), ('and', 'cc', 'unknown'), ('then', 'rb', 'unknown'), ('leave', 'vb', 'unknown'), ('russia', 'nn', 'unknown'), ('?', '.', 'unknown')]
```
Word features are further augmented with extra information such as, Is it the beginning of sentence, word ending syllable, etc.
```
['bias', 'word.lower=how', 'word[-3:]=how', 'word[-2:]=ow', 'word.isupper=False', 'word.istitle=False', 'word.isdigit=False', 'postag=wrb', 'postag[:2]=wr', 'BOS', '+1:word.lower=did', '+1:word.istitle=False', '+1:word.isupper=False', '+1:postag=vbd', '+1:postag[:2]=vb']
```

Dataset is divided into 5 folds for cross-validation and model selection. Each fold preserves the average percentage of classes.

```
============training for fold 2 ...============

Training Naive Bayes Classifier.
==================================
Testing Naive Bayes.
==================================
Accuracy: 0.925675675676
Confusion matrix, without normalization
[[ 21   0   0   0   0]
 [  1  47   2   2   2]
 [  0   5 110   4   3]
 [  0   1   0  18   0]
 [  0   0   0   2  78]]


Training Maxent Classifier. Press Ctrl+C to stop training and see model predition.
==================================================================================
^C      Training stopped: keyboard interrupt
Testing Maxent Classifier.
==========================
Accuracy: 0.699324324324
Confusion matrix, without normalization
[[  0   3  18   0   0]
 [  0  28  25   0   1]
 [  0   0 120   0   2]
 [  0   0  19   0   0]
 [  0   0  21   0  59]]


CRF Training Parameters
=======================
['feature.minfreq', 'feature.possible_states', 'feature.possible_transitions', 'c1', 'c2', 'max_iterations', 'num_memories', 'epsilon', 'period', 'delta', 'linesearch', 'max_linesearch']


Random sentence from test set
=========================

when was cnn 's first broadcast ?

('Predicted:', 'when when when when when when when')
('Correct:  ', 'when when when when when when when')

             precision    recall  f1-score   support

affirmation       1.00      1.00      1.00       175
    unknown       0.98      0.93      0.96       569
       what       0.97      0.98      0.97      1237
       when       0.95      0.83      0.89       181
        who       0.96      1.00      0.98       754

avg / total       0.97      0.97      0.97      2916

```
This shows the output of train.py for second fold of dataset. We can observe that CRF clearly outperforms NaiveBayes and MaxEnt classifier for sentence labelling task.

Note: MaxEnt classifier taskes maximum time to train, so the training can be stopped midway and the models are saved. It performs relatively well when its trained for around 1000 iterations.


For further explanation and help, please drop a mail at:
contactbrijmohan@gmail.com
