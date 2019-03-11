#_*_coding:utf-8_*_
from sklearn.cluster import KMeans
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import gensim
import codecs
from gensim.models.doc2vec import TaggedDocument

model = Doc2Vec.load('../data/novel.doc2vec')
file = codecs.open('../data/novel.txt','r','utf8')

train = []
for i,text in enumerate(file):
    docu = TaggedDocument(text,[i])
    train.append(docu)

vectors_list = []
for texts,label in train:
    vec = model.infer_vector(text)
    vectors_list.append(vec)

kmeans_model = KMeans(n_clusters=15)
kmeans_model.fit(vectors_list)
labels = kmeans_model.predict(vectors_list[1:100])
# cluster_center= kmeans_model.cluster_centers_

with codecs.open("../data/own_classify.txt", 'a','utf8') as wf:
    for i in range(99):
        string = ""
        text = train[i][0]
        for word in text:
            string = string + word
        string = string + "\t"
        string = string + str(labels[i])
        string = string + '\n'
        wf.write(string)

# print(cluster_center)
