from pyforest import *
import os
import nltk
import spacy
os.chdir(r'C:\Users\bishw\OneDrive\Desktop')
file= open('34.2 gadsby_manuscript.txt','r',encoding="utf8")
nlp=spacy.load('en_core_web_sm')
nltk.download('punkt')
from gensim.utils import tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction import text
# doc=nlp(file.read())
# sent=[]
# for word in doc:
#     if not word.is_punct | word.is_space:
#       sent.append(word.text)
# print(sent)
texts=file.read(
#convert each sentence into lists using nltk senstence tokenizer
sent=nltk.sent_tokenize(texts)
corpus=[]
for i in sent:
# tokenizing using gensim it also removes punctuations and spaces if any
    corpus.append(list(tokenize(i, lowercase=True)))
print(corpus) #this is a bag of sentences
#using W2V on the sentences just like in tfidf or countvec
w2v = Word2Vec(corpus, min_count=1)
words=w2v.wv.vocab
#checking vector of all
vector=w2v.wv['all']
print(vector)
#checking similar words w.r.t all
similar=w2v.wv.most_similar('all')
print(similar)