# importing modules
import nltk
import spacy
import re
from gensim.models.fasttext import FastText
from nltk import sent_tokenize
from nltk import  word_tokenize
from nltk.corpus import stopwords
from gensim.utils import tokenize
import os
from sklearn.decomposition import PCA
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\bishw\OneDrive\Desktop')
document=open('34.2 gadsby_manuscript.txt', encoding='utf-8')
document=document.read()
# stpwrds=list(stopwords.words('english'))
# rmr=['not','no']
# for i in rmr:
#     stpwrds.pop(stpwrds.index(i))
# sentence=sent_tokenize(document)
# word=[word_tokenize(i) for i in sentence]
# word=[w for w in word[0] if w not in stpwrds]
#
# word=[re.sub('[.,!@#$%&:;`’”]','',i) for i in word]
# for i in word:
#     if i=='':
#         word.pop(word.index(i))
# word.pop(-1)
# word.pop(-3)
# word.pop(-4)
# word.pop(19)

sentence=sent_tokenize(document)
#print(sentence)
corpus=[]
for i in sentence:
    corpus.append(list(tokenize(i, lowercase=True)))

fasttext_model = FastText(corpus,
                          size=100,
                          window=30,
                          min_count=5,
                          sample=1e-3,
                          sg=1, # sg decides whether to use the skip-gram model (1) or CBOW (0)
                          iter=50)
# print(fasttext_model.wv.vocab)
# print(fasttext_model.wv['youth'])
# print(fasttext_model.wv.most_similar(positive='youth'))
# print(fasttext_model.wv.similarity(w1='youth', w2='youthful'))
# print(fasttext_model.wv.similarity(w1='youth', w2='child'))
cluster1=fasttext_model.wv.most_similar(positive='child', topn=15)
cluster2=fasttext_model.wv.most_similar(positive='youth', topn=15)
# print(cluster1)
# print(cluster2)
pca=PCA(n_components=2)
c1={}
for k,v in cluster1:
    c1[k]=fasttext_model.wv[k]
c2={}
for k,v in cluster2:
    c2[k]=fasttext_model.wv[k]

df=pd.DataFrame(c1).transpose().reset_index()
dft=pd.DataFrame(pca.fit_transform(df.iloc[:,1:]), index=df['index'],columns=['x','y'])
df1=pd.DataFrame(c2).transpose().reset_index()
dft1=pd.DataFrame(pca.fit_transform(df1.iloc[:,1:]), index=df1['index'],columns=['x','y'])
# print(dft)
# print(dft1)
df=pd.concat([dft,dft1], axis=0)
# print(df)

f, axes = plt.subplots(1,1 , figsize=(50,50), sharex=True, sharey=True)
sns.scatterplot(df.x, df.y)
for word, pos in df.iterrows():
    axes.annotate(word, pos)
