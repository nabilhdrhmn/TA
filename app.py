from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText as ft
from gensim.similarities import WmdSimilarity

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

pd.options.mode.chained_assignment = None  # default='warn
# app = Flask(__name__)
app = Flask(__name__,
            static_url_path='',
            static_folder='public',
            template_folder='templates')


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

@app.route('/index_cocomo')
def index_cocomo():
    return render_template('index_cocomo.html')

@app.route('/index_pm')
def index_pm():
    return render_template('index_pm.html')

@app.route('/index_ss')
def index_ss():
    return render_template('index_ss.html')

@app.route('/index_mr')
def index_mr():
    return render_template('index_mr.html')

@app.route('/cocomo', methods=['POST', 'GET'])
def cocomo():
    if request.method == "POST":
        try:
            sentence = request.form['query']
        except:
            errors.append(
                "Can't read!"
                )
            return render_template('cocomo.html', errors=errors)
        if sentence:
            df = pd.read_excel('Evaluation.xlsx')
            df_test = pd.read_excel('Test.xlsx')
            test = df_test['Title'].to_list()
            df['content'] = df[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 
            for x in range(128):
                    if df.Title.iloc[x] in test:
                        df.Label.iloc[x] = 'Yes'
                    else:
                        df.Label.iloc[x] = 'No'
            cocomo_df = df[df['Label']=='Yes']
            cocomo_content = cocomo_df['content'].to_list()
            cocomo_corpus = list()
            cocomo_lemma = list()
            for x in cocomo_content:
                    cocomo_corpus.append(preprocess(x))
            for x in cocomo_corpus:
                    cocomo_lemma.append(lemmatize(x))
            model_ft = ft(cocomo_lemma, sg=1, workers=3, iter=5,size=100, min_count=5, window=2)
            instance_ft = WmdSimilarity(cocomo_lemma, model_ft, num_best=105)
            query = sentence
            query = preprocess(query)
            query = lemmatize(query)
            sims_ft = instance_ft[query]
            wmd = list()
            tp = 0
            fn = 0
            for i in range(105):
                if round(sims_ft[i][1],2) >= 0.85:
                    wmd.append(cocomo_df.Title.iloc[sims_ft[i][0]])
                    tp = tp + 1
                else:
                    fn = fn + 1
            df_tp = pd.DataFrame()
            for metadata in wmd:
                    df_tp=df_tp.append(df[df['Title']==metadata])
            df_tp = df_tp.drop(['Abstract', 'Keywords','Label','content'], axis=1)
    return render_template('cocomo.html', tables=[df_tp.to_html()])

@app.route('/pm', methods=['POST', 'GET'])
def pm():
    if request.method == "POST":
        try:
            sentence_pm = request.form['query_pm']
        except:
            errors.append(
                "Can't read!"
                )
            return render_template('pm.html', errors=errors)
        if sentence_pm:
            df_pm = pd.read_excel('Process Mining.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])
            df_pm['content'] = df_pm[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1)
            pm = df_pm[df_pm['Label']=='Yes']
            del pm['Label']
            content_pm = pm['content'].to_list()
            pm_corpus = list()
            pm_lemma = list()
            for x in content_pm:
                pm_corpus.append(preprocess(x))
            for x in pm_corpus:
                pm_lemma.append(lemmatize(x))
            model_ft_pm = ft(pm_lemma, sg=1, workers=3, iter=5,size=100, min_count=5, window=2)
            instance_ft_pm = WmdSimilarity(pm_lemma, model_ft_pm, num_best=105)
            query_pm = sentence_pm
            query_pm = preprocess(query_pm)
            query_pm = lemmatize(query_pm)
            sims_ft_pm = instance_ft_pm[query_pm]
            wmd = list()
            tp = 0
            fn = 0
            for i in range(len(pm_lemma)):
                if round(sims_ft_pm[i][1],2) >= 0.75:
                    wmd.append(pm.Title.iloc[sims_ft_pm[i][0]])
                    tp = tp + 1
                else:
                    fn = fn + 1
            df_tp = pd.DataFrame()
            for metadata in wmd:
                    df_tp=df_tp.append(pm[pm['Title']==metadata])
            df_tp = df_tp.drop(['Abstract', 'Keywords', 'content'], axis=1)
    return render_template('pm.html', tables=[df_tp.to_html()])

@app.route('/ss', methods=['POST', 'GET'])
def ss():
    if request.method == "POST":
        try:
            sentence_ss = request.form['query_ss']
        except:
            errors.append(
                "Can't read!"
                )
            return render_template('ss.html', errors=errors)
        if sentence_ss:
            df_ss = pd.read_excel('Semantic Search.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])
            df_ss['content'] = df_ss[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 
            ss = df_ss[df_ss['Label']=='Yes']
            del ss['Label']
            content_ss = ss['content'].to_list()
            ss_corpus = list()
            ss_lemma = list()
            for x in content_ss:
                ss_corpus.append(preprocess(x))
            for x in ss_corpus:
                ss_lemma.append(lemmatize(x))
            model_ft_ss = ft(ss_lemma, sg=1, workers=3, iter=5,size=100, min_count=5, window=2)
            instance_ft_ss = WmdSimilarity(ss_lemma, model_ft_ss, num_best=105)
            query_ss = sentence_ss
            query_ss = preprocess(query_ss)
            query_ss = lemmatize(query_ss)
            sims_ft_ss = instance_ft_ss[query_ss]
            wmd = list()
            tp = 0
            fn = 0
            for i in range(len(ss_lemma)):
                if round(sims_ft_ss[i][1],2) >= 0.75:
                    wmd.append(ss.Title.iloc[sims_ft_ss[i][0]])
                    tp = tp + 1
                else:
                    fn = fn + 1
            df_tp = pd.DataFrame()
            for metadata in wmd:
                    df_tp=df_tp.append(ss[ss['Title']==metadata])
            df_tp = df_tp.drop(['Abstract', 'Keywords', 'content'], axis=1)
    return render_template('ss.html', tables=[df_tp.to_html()])

@app.route('/mr', methods=['POST', 'GET'])
def mr():
    if request.method == "POST":
        try:
            sentence_mr = request.form['query_mr']
        except:
            errors.append(
                "Can't read!"
                )
            return render_template('mr.html', errors=errors)
        if sentence_mr:
            df_mr = pd.read_excel('Mixed Reality.xlsx', names=["Title", "Abstract", "Keywords", "Source", "Label"])
            df_mr['content'] = df_mr[['Title', 'Abstract', 'Keywords']].apply(lambda x: ' '.join(x), axis = 1) 
            mr = df_mr[df_mr['Label']=='Yes']
            del mr['Label']
            content_mr = mr['content'].to_list()
            mr_corpus = list()
            mr_lemma = list()
            for x in content_mr:
                mr_corpus.append(preprocess(x))
            for x in mr_corpus:
                mr_lemma.append(lemmatize(x))
            model_ft_mr = ft(mr_lemma, sg=1, workers=3, iter=5,size=100, min_count=5, window=2)
            instance_ft_mr = WmdSimilarity(mr_lemma, model_ft_mr, num_best=105)
            query_mr = sentence_mr
            query_mr = preprocess(query_mr)
            query_mr = lemmatize(query_mr)
            sims_ft_mr = instance_ft_mr[query_mr]
            wmd = list()
            tp = 0
            fn = 0
            for i in range(len(mr_lemma)):
                if round(sims_ft_mr[i][1],2) >= 0.77:
                    wmd.append(mr.Title.iloc[sims_ft_mr[i][0]])
                    tp = tp + 1
                else:
                    fn = fn + 1
            df_tp = pd.DataFrame()
            for metadata in wmd:
                    df_tp=df_tp.append(mr[mr['Title']==metadata])
            df_tp = df_tp.drop(['Abstract', 'Keywords','content'], axis=1)
    return render_template('mr.html', tables=[df_tp.to_html()])

if __name__ == '__main__':
    app.run(debug=True)


