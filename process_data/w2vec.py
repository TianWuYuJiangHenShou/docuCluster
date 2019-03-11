#_*_coding:utf-8_*_
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

new = codecs.open('../data/novel.txt','r','utf8')
# model = Word2Vec(LineSentence(new),sg=0,size=192,window=5,min_count=5,workers=multiprocessing.cpu_count()-1)
# model.save('../data/novel.word2vec')


train = []
for i,text in enumerate(new):
    docu = TaggedDocument(text,[i])
    train.append(docu)

model= Doc2Vec(train,min_count=1,window=3,vectors_size=200,sample=1e-3,negative=5,workers=4)
model.save('../data/novel.doc2vec')