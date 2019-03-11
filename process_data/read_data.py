#_*_coding:utf-8_*_

import pandas as pd
import codecs
from sklearn.cluster import KMeans
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
data = pd.read_excel('../data/raw.xlsx')
text = []
for line in data['summary']:
    text.append(str(line))

vec = []
stop_list = []


with codecs.open('../data/stopword.txt','r',encoding='utf8')as f:
    read = f.read().splitlines()
    for word in read:
        stop_list.append(word)

for sen in text:
    filter_list = []
    sen = list(jieba.cut(sen))
    for w in sen:
        if w not in stop_list:
            filter_list.append(w)
    vec.append(filter_list)

file = codecs.open('../data/novel.txt','a','utf8')
for line in vec:
    tmp = ' '.join(line)
    file.write(tmp+'\n')

