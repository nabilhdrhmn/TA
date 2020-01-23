#!/usr/bin/env python
# coding: utf-8

# In[334]:


import pandas as pd
import math


# In[266]:


threshold = 1


# In[39]:


df_pm = pd.read_excel('Process Mining.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])
df_ss = pd.read_excel('Semantic Search.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])
df_mr = pd.read_excel('Mixed Reality.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])


# In[40]:


df_pm['content'] = df_pm[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 
df_ss['content'] = df_ss[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 
df_mr['content'] = df_mr[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 


# In[41]:


pm = df_pm[df_pm['Label']=='Yes']
del pm['Label']
non_pm = df_pm[df_pm['Label']=='No']
del non_pm['Label']

ss = df_ss[df_ss['Label']=='Yes']
del ss['Label']
non_ss = df_ss[df_ss['Label']=='No']
del non_ss['Label']

mr = df_mr[df_mr['Label']=='Yes']
del mr['Label']
non_mr = df_mr[df_mr['Label']=='No']
del non_mr['Label']


# In[43]:


content_pm = pm['content'].to_list()
content_non_pm = non_pm['content'].to_list()
content_ss = ss['content'].to_list()
content_non_ss = non_ss['content'].to_list()
content_mr = mr['content'].to_list()
content_non_mr = non_mr['content'].to_list()


# In[44]:


len(content_pm)


# In[45]:


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc

def lemmatize(doc):
    doc_lemma = list()
    for word in doc:
        doc_lemma.append(lemmatizer.lemmatize(word))
    return doc_lemma


# In[46]:


pm_corpus = list()
pm_lemma = list()
for x in content_pm:
    pm_corpus.append(preprocess(x))
for x in pm_corpus:
    pm_lemma.append(lemmatize(x))


# In[47]:


non_pm_corpus = list()
non_pm_lemma = list()
for x in content_non_pm:
    non_pm_corpus.append(preprocess(x))
for x in non_pm_corpus:
    non_pm_lemma.append(lemmatize(x))


# In[48]:


ss_corpus = list()
ss_lemma = list()
for x in content_ss:
    ss_corpus.append(preprocess(x))
for x in ss_corpus:
    ss_lemma.append(lemmatize(x))


# In[49]:


non_ss_corpus = list()
non_ss_lemma = list()
for x in content_non_ss:
    non_ss_corpus.append(preprocess(x))
for x in non_ss_corpus:
    non_ss_lemma.append(lemmatize(x))


# In[50]:


mr_corpus = list()
mr_lemma = list()
for x in content_mr:
    mr_corpus.append(preprocess(x))
for x in mr_corpus:
    mr_lemma.append(lemmatize(x))


# In[51]:


non_mr_corpus = list()
non_mr_lemma = list()
for x in content_non_mr:
    non_mr_corpus.append(preprocess(x))
for x in non_mr_corpus:
    non_mr_lemma.append(lemmatize(x))


# In[52]:


import multiprocessing
from time import time

multiprocessing.cpu_count()


# In[53]:


from gensim.models import FastText as ft


# In[273]:


start = time()
model_ft_pm = ft(pm_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[274]:


start = time()
model_ft_non_pm = ft(non_pm_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[275]:


start = time()
model_ft_ss = ft(ss_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[276]:


start = time()
model_ft_non_ss = ft(non_ss_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[277]:


start = time()
model_ft_mr = ft(mr_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[278]:


start = time()
model_ft_non_mr = ft(non_mr_lemma, sg=1, workers=multiprocessing.cpu_count()-1, size=100, iter=5, min_count=5, window=2)
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[60]:


from gensim.similarities import WmdSimilarity


# In[172]:


threshold_ss_1 = 0.85
threshold_ss_2 = 0.85
threshold_ss_3 = 0.85
threshold_pm_1 = 0.85
threshold_pm_2 = 0.85
threshold_pm_3 = 0.85
threshold_mr_1 = 0.85
threshold_mr_2 = 0.85
threshold_mr_3 = 0.85


# In[173]:


threshold_non_ss_1 = 0.85
threshold_non_ss_2 = 0.85
threshold_non_ss_3 = 0.85
threshold_non_pm_1 = 0.85
threshold_non_pm_2 = 0.85
threshold_non_pm_3 = 0.85
threshold_non_mr_1 = 0.85
threshold_non_mr_2 = 0.85
threshold_non_mr_3 = 0.85


# In[279]:


instance_ft_pm = WmdSimilarity(pm_lemma, model_ft_pm, num_best=len(pm_lemma))
instance_ft_non_pm = WmdSimilarity(non_pm_lemma, model_ft_non_pm, num_best=len(non_pm_lemma))

instance_ft_ss = WmdSimilarity(ss_lemma, model_ft_ss, num_best=len(ss_lemma))
instance_ft_non_ss = WmdSimilarity(non_ss_lemma, model_ft_non_ss, num_best=len(non_ss_lemma))

instance_ft_mr = WmdSimilarity(mr_lemma, model_ft_mr, num_best=len(mr_lemma))
instance_ft_non_mr = WmdSimilarity(non_mr_lemma, model_ft_non_mr, num_best=len(non_mr_lemma))


# ## Skenario 1

# In[368]:


start = time()
sentence_pm = 'process mining'
query = preprocess(sentence_pm)
query_pm = lemmatize(query)
sims_ft_pm = instance_ft_pm[query_pm]
sims_ft_non_pm = instance_ft_non_pm[query_pm]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[399]:


start = time()
sentence_ss = 'semantic search'
query = preprocess(sentence_ss)
query_ss = lemmatize(query)
sims_ft_ss = instance_ft_ss[query_ss]
sims_ft_non_ss = instance_ft_non_ss[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[363]:


start = time()
sentence_mr = 'mixed reality'
query = preprocess(sentence_mr)
query_mr = lemmatize(query)
sims_ft_mr = instance_ft_mr[query_mr]
sims_ft_non_mr = instance_ft_non_mr[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[383]:


similar = model_ft_pm.most_similar(positive=query_pm, topn=400)
pm_most_similar_1 = similar[0][0]
pm_most_similar_10 = similar[9][0]
pm_most_similar_20 = similar[19][0]
# pm_most_similar_1
a = list()
for x in range(len(similar)):
    a.append(similar[x][0])
    
'techniques' in a


# In[398]:


similar = model_ft_ss.most_similar(positive=query_ss, topn=50)
ss_most_similar_1 = similar[0][0]
ss_most_similar_10 = similar[9][0]
ss_most_similar_20 = similar[19][0]
# ss_most_similar_20

a = list()
for x in range(len(similar)):
    a.append(similar[x][0])
    
'engine' in a


# In[411]:


similar = model_ft_mr.most_similar(positive=query_mr, topn=50)
mr_most_similar_1 = similar[0][0]
mr_most_similar_10 = similar[9][0]
mr_most_similar_20 = similar[19][0]
# mr_most_similar_20

a = list()
for x in range(len(similar)):
    a.append(similar[x][0])

'realization' in a


# In[372]:


wmd = list()
print('Query:')
print(sentence_pm)
tp = 0
fn = 0
for i in range(len(pm_lemma)):
    if round(sims_ft_pm[i][1],2) >= threshold_pm_1:
        wmd.append(pm.Title.iloc[sims_ft_pm[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
print("Hasil True Positive Process Mining: {}".format(tp))
print("Hasil False Negative Process Mining: {}".format(fn))


# In[289]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_pm[df_pm['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[374]:


print('Query:')
print(sentence_pm)
fp = 0
tn = 0
for i in range(len(non_pm_lemma)):
    if round(sims_ft_non_pm[i][1],2) >= threshold_non_pm_1:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Process Mining: {}".format(tn))
print("Hasil False Positive Process Mining: {}".format(fp))


# In[209]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 1 Process Mining {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 1 Process Mining {}".format(precision))
print("Hasil Recall dari Skenario 1 Process Mining {}".format(recall))
print("Hasil F-1 Score dari Skenario 1 Process Mining {}".format(f1_score))


# In[400]:


wmd = list()
print('Query:')
print(sentence_ss)
tp = 0
fn = 0
for i in range(len(ss_lemma)):
    if round(sims_ft_ss[i][1],2) >= threshold_ss_1:
        wmd.append(ss.Title.iloc[sims_ft_ss[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Semantic Search: {}".format(tp))
print("Hasil False Negative Semantic Search: {}".format(fn))


# In[212]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_ss[df_ss['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[216]:


print('Query:')
print(sentence_ss)
tn = 0
fp = 0
for i in range(len(non_ss_lemma)):
    if sims_ft_non_ss[i][1] > threshold_non_ss_1:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Semantic Search: {}".format(c))
print("Hasil False Positive Semantic Search: {}".format(count))


# In[218]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 1 Semantic Search {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 1 Semantic Search {}".format(precision))
print("Hasil Recall dari Skenario 1 Semantic Search {}".format(recall))
print("Hasil F-1 Score dari Skenario 1 Semantic Search {}".format(f1_score))


# In[220]:


wmd = list()
print('Query:')
print(sentence_mr)
tp = 0
fn = 0
for i in range(len(mr_lemma)):
    if sims_ft_mr[i][1] > threshold_mr_1:
        wmd.append(mr.Title.iloc[sims_ft_mr[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Mixed Reality: {}".format(tp))
print("Hasil False Negative Mixed Reality: {}".format(fn))


# In[221]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_mr[df_mr['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[224]:


print('Query:')
print(sentence_mr)
tn = 0
fp = 0
for i in range(len(non_mr_lemma)):
    if sims_ft_non_mr[i][1] > threshold_non_mr_1:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Mixed Reality: {}".format(tn))
print("Hasil False Positive Mixed Reality: {}".format(fp))


# In[225]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 1 Mixed Reality {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 1 Mixed Reality {}".format(precision))
print("Hasil Recall dari Skenario 1 Mixed Reality {}".format(recall))
print("Hasil F-1 Score dari Skenario 1 Mixed Reality {}".format(f1_score))


# ## Skenario 2

# In[387]:


start = time()
sentence_pm = 'process mining' + ' techniques ' + 'model'
query = preprocess(sentence_pm)
query_pm = lemmatize(query)
sims_ft_pm = instance_ft_pm[query_pm]
sims_ft_non_pm = instance_ft_non_pm[query_pm]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[403]:


start = time()
sentence_ss = 'semantic search' + ' query'
query = preprocess(sentence_ss)
query_ss = lemmatize(query)
sims_ft_ss = instance_ft_ss[query_ss]
sims_ft_non_ss = instance_ft_non_ss[query_ss]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[293]:


start = time()
sentence_mr = 'mixed reality' + ' virtual'
query = preprocess(sentence_mr)
query_mr = lemmatize(query)
sims_ft_mr = instance_ft_mr[query]
sims_ft_non_mr = instance_ft_non_mr[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[388]:


wmd = list()
print('Query:')
print(sentence_pm)
tp = 0
fn = 0
for i in range(len(pm_lemma)):
    if round(sims_ft_pm[i][1],2) >= threshold_pm_1:
        wmd.append(pm.Title.iloc[sims_ft_pm[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
print("Hasil True Positive Process Mining: {}".format(tp))
print("Hasil False Negative Process Mining: {}".format(fn))


# In[343]:


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# In[346]:


truncate(sims_ft_pm[0][1], 4)


# In[231]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_pm[df_pm['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[233]:


print('Query:')
print(sentence_pm)
tn = 0
fp = 0
for i in range(len(non_pm_lemma)):
    if sims_ft_non_pm[i][1] > threshold_non_pm_2:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Process Mining: {}".format(tn))
print("Hasil False Positive Process Mining: {}".format(fp))


# In[234]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 2 Process Mining {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 2 Process Mining {}".format(precision))
print("Hasil Recall dari Skenario 2 Process Mining {}".format(recall))
print("Hasil F-1 Score dari Skenario 2 Process Mining {}".format(f1_score))


# In[405]:


wmd = list()
print('Query:')
print(sentence_ss)
tp = 0
fn = 0
for i in range(len(ss_lemma)):
    if round(sims_ft_ss[i][1],2) >= .75:
        wmd.append(ss.Title.iloc[sims_ft_ss[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Semantic Search: {}".format(tp))
print("Hasil False Negative Semantic Search: {}".format(fn))


# In[237]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_ss[df_ss['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[238]:


print('Query:')
print(sentence_ss)
tn = 0
fp = 0
for i in range(len(non_ss_lemma)):
    if sims_ft_non_ss[i][1] > threshold_non_ss_2:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Semantic Search: {}".format(tn))
print("Hasil False Positive Semantic Search: {}".format(fp))


# In[239]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 2 Semantic Search {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 2 Semantic Search {}".format(precision))
print("Hasil Recall dari Skenario 2 Semantic Search {}".format(recall))
print("Hasil F-1 Score dari Skenario 2 Semantic Search {}".format(f1_score))


# In[241]:


wmd = list()
print('Query:')
print(sentence_mr)
tp = 0
fn = 0
for i in range(len(mr_lemma)):
    if sims_ft_mr[i][1] > threshold_mr_2:
        wmd.append(mr.Title.iloc[sims_ft_mr[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Mixed Reality: {}".format(tp))
print("Hasil False Negative Mixed Reality: {}".format(fn))


# In[242]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_mr[df_mr['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[244]:


print('Query:')
print(sentence_mr)
tn = 0
fp = 0
for i in range(len(non_mr_lemma)):
    if sims_ft_non_mr[i][1] > threshold_non_mr_2:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Mixed Reality: {}".format(tn))
print("Hasil False Positive Mixed Reality: {}".format(fp))


# In[245]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 2 Mixed Reality {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 2 Mixed Reality {}".format(precision))
print("Hasil Recall dari Skenario 2 Mixed Reality {}".format(recall))
print("Hasil F-1 Score dari Skenario 2 Mixed Reality {}".format(f1_score))


# ## Skenario 3

# In[389]:


start = time()
sentence_pm = 'process mining '+ pm_most_similar_1
query = preprocess(sentence)
query_pm = lemmatize(query)
sims_ft_pm = instance_ft_pm[query_pm]
sims_ft_non_pm = instance_ft_non_pm[query_pm]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[297]:


start = time()
sentence_ss = 'semantic search' + ss_most_similar
query = preprocess(sentence)
query_ss = lemmatize(query)
sims_ft_ss = instance_ft_ss[query]
sims_ft_non_ss = instance_ft_non_ss[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[298]:


start = time()
sentence_mr = 'mixed reality' + mr_most_similar
query = preprocess(sentence)
query_mr = lemmatize(query)
sims_ft_mr = instance_ft_mr[query]
sims_ft_non_mr = instance_ft_non_mr[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[390]:


wmd = list()
print('Query:')
print(sentence_pm)
tp = 0
fn = 0
for i in range(len(pm_lemma)):
    if round(sims_ft_pm[i][1],2) >= .75:
        wmd.append(pm.Title.iloc[sims_ft_pm[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
print("Hasil True Positive Process Mining: {}".format(tp))
print("Hasil False Negative Process Mining: {}".format(fn))


# In[248]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_pm[df_pm['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[249]:


print('Query:')
print(sentence_pm)
tn = 0
fp = 0
for i in range(len(non_pm_lemma)):
    if sims_ft_non_pm[i][1] > threshold_non_pm_3:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Process Mining: {}".format(c))
print("Hasil False Positive Process Mining: {}".format(count))


# In[250]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 3 Process Mining {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 3 Process Mining {}".format(precision))
print("Hasil Recall dari Skenario 3 Process Mining {}".format(recall))
print("Hasil F-1 Score dari Skenario 3 Process Mining {}".format(f1_score))


# In[252]:


wmd = list()
print('Query:')
print(sentence_ss)
tp = 0
fn = 0
for i in range(len(ss_lemma)):
    if sims_ft_ss[i][1] > threshold_ss_3:
        wmd.append(ss.Title.iloc[sims_ft_ss[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Semantic Search: {}".format(tp))
print("Hasil False Negative Semantic Search: {}".format(fn))


# In[253]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_ss[df_ss['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[255]:


print('Query:')
print(sentence_ss)
tn = 0
fp = 0
for i in range(len(non_ss_lemma)):
    if sims_ft_non_ss[i][1] > threshold_non_ss_3:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Semantic Search: {}".format(tn))
print("Hasil False Positive Semantic Search: {}".format(fp))


# In[256]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 3 Semantic Search {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 3 Semantic Search {}".format(precision))
print("Hasil Recall dari Skenario 3 Semantic Search {}".format(recall))
print("Hasil F-1 Score dari Skenario 3 Semantic Search {}".format(f1_score))


# In[258]:


wmd = list()
print('Query:')
print(sentence_mr)
tp = 0
fn = 0
for i in range(len(mr_lemma)):
    if sims_ft_mr[i][1] > threshold_mr_3:
        wmd.append(mr.Title.iloc[sims_ft_mr[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
        pass
print("Hasil True Positive Mixed Reality: {}".format(tp))
print("Hasil False Negative Mixed Reality: {}".format(fn))


# In[259]:


df_tp = pd.DataFrame()
for metadata in wmd:
    df_tp=df_tp.append(df_mr[df_mr['Title']==metadata])

df_tp = df_tp.drop(['Label'], axis=1)
df_tp.head()


# In[260]:


print('Query:')
print(sentence_mr)
tn = 0
fp = 0
for i in range(len(non_mr_lemma)):
    if sims_ft_non_mr[i][1] > threshold_non_mr_3:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Mixed Reality: {}".format(tn))
print("Hasil False Positive Mixed Reality: {}".format(fp))


# In[261]:


cohen_kappa = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2 * (precision * recall)/(precision + recall)
print("Hasil Cohen's Kappa dari Skenario 3 Mixed Reality {}".format(cohen_kappa))
print("Hasil Precision dari Skenario 3 Mixed Reality {}".format(precision))
print("Hasil Recall dari Skenario 3 Mixed Reality {}".format(recall))
print("Hasil F-1 Score dari Skenario 3 Mixed Reality {}".format(f1_score))


# In[ ]:


start = time()
sentence = 'process mining'
query = preprocess(sentence)
query_pm = lemmatize(query)
sims_ft_pm = instance_ft_pm[query]
sims_ft_non_pm = instance_ft_non_pm[query]
print('Cell took %.2f minutes to run.' %((time() - start)/60))


# In[ ]:


wmd = list()
print('Query:')
print(sentence_pm)
tp = 0
fn = 0
for i in range(len(pm_lemma)):
    if sims_ft_pm[i][1] > threshold:
        wmd.append(pm.Title.iloc[sims_ft_pm[i][0]])
        tp = tp + 1
    else:
        fn = fn + 1
print("Hasil True Positive Process Mining: {}".format(tp))
print("Hasil False Negative Process Mining: {}".format(fn))


# In[ ]:


print('Query:')
print(sentence_pm)
fp = 0
tn = 0
for i in range(len(non_pm_lemma)):
    if sims_ft_non_pm[i][1] > threshold:
        fp = fp + 1
    else:
        tn = tn + 1
        pass
print("Hasil True Negative Process Mining: {}".format(tn))
print("Hasil False Positive Process Mining: {}".format(fp))

