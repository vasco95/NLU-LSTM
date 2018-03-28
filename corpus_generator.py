#!/usr/bin/python

# This file combines all the documents in a given corpus to give a single mega
# file. Additionally a for each sentence SOS and EOS are added.
#
import re
import nltk
import string
import os, sys
import numpy as np
from nltk.corpus import brown
from nltk.corpus import gutenberg

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('latin-1')

# SOS0 = '<sos>'
SOS0 = ''
EOS0 = '<eos>'

prefix = SOS0 #+ ' ' + SOS0
postfix = EOS0 #+ ' ' + EOS1
unknown = '<unk>'

def create_corpus(dir_name = 'brown'):
    corpus = brown
    if dir_name == 'gutenberg':
        corpus = gutenberg

    sentences = []
    cnt = 0
    avgr = 0
    for entry in corpus.sents():
        stmp = ''
        avgr += len(entry)
        for word in entry:
            if word.isdigit() == False:
                # stmp += word.lower().strip(string.punctuation) + ' '
                stmp += word.lower() + ' '
        sentences.append(stmp);
        cnt += 1
        if cnt >= 10000:
            break
    print avgr * 1.0 / cnt

    idx = np.random.randint(len(sentences), size = len(sentences))
    train_len = int(len(sentences) * 0.8)
    devset_len = int(len(sentences) * 0.1)

    fout = open(dir_name + '.train', 'w')
    for ii in range(train_len):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)
    # fout.write(' ' + prefix + ' ' + unknown + ' ' + postfix + ' ')
    fout.close()

    fout = open(dir_name + '.dev', 'w')
    for ii in range(train_len, (train_len + devset_len)):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)
    fout.close()

    fout = open(dir_name + '.test', 'w')
    for ii in range(devset_len + train_len, len(sentences)):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)
    fout.close()

create_corpus('gutenberg')
print 'Created gutenberg'
