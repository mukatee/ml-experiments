__author__ = 'teemu kanstren'

#simple predictor using pre-trained LGBM classifier.

import pandas as pd
import numpy as np
import pickle
import re
import gensim
from sklearn.externals import joblib

from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from string import punctuation

print("collecting stopwords")

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(punctuation))
stop_words.update(["''", "--", "**"])

print("loading bugs")

df_2019 = pd.read_csv("bugs2019/reduced.csv", parse_dates=["Created", "Due Date", "Resolved"])
df_2019.dropna(subset=['comp1', "Description"], inplace=True)
df_2019 = df_2019.reset_index()

def preprocess_report_part1(body_o):
    # replace all but alphabetical and numerical characters and some specials such as /\
    body = re.sub('[^A-Za-z0-9 /\\\_+.,:\n]+', '', body_o)
    # replace URL separators with space so the parts of the url become separate words
    body = re.sub('[/\\\]', ' ', body)
    # finally lemmatize all words for the analysis
    lemmatizer = WordNetLemmatizer()
    # text tokens are basis for the features
    text_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(body.lower()) if word not in stop_words]
    return text_tokens

features = df_2019["Description"]

print("processing descriptions")

features2 = features.apply(lambda x: preprocess_report_part1(x))

token_lists = list(features2)

print("bi- and tri-gramming")

common_terms = ["of", "with", "without", "and", "or", "the", "a"]
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Build the bigram and trigram models
bigram = gensim.models.Phrases(token_lists, min_count=5, threshold=100, common_terms=common_terms) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[token_lists], threshold=100, common_terms=common_terms)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


print("loading label encoder")

#https://stackoverflow.com/questions/28656736/using-scikits-labelencoder-correctly-across-multiple-programs
encoder = LabelEncoder()
encoder.classes_ = np.load('le_classes.npy')

le_id_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

print("loading tfidf vectorizer")

vectorizer = pickle.load( open( "tfidf.pickle", "rb" ) )

print("loading mnb")

# load model
model = joblib.load('multin-nb.mdl')

print("predicting")


def predict(bug_description):
    features = preprocess_report_part1(bug_description)
    token_lists = list(features)
    clubbed_tokens = [trigram_mod[bigram_mod[text]] for text in token_lists]
    texts = [" ".join(tokens) for tokens in clubbed_tokens]
    features_transformed = vectorizer.transform(texts)

    probs = model.predict_proba(features_transformed)
    result = []
    for idx in range(probs.shape[1]):
        name = le_id_mapping[idx]
        prob = (probs[0, idx]*100)
        prob_str = "%.2f%%" % prob
        #print(name, ":", prob_str)
        result.append((name, prob))
    return result

def predict_file():
    with open('example_bug.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')

    result = predict(data)
    sorted_by_second = sorted(result, key=lambda x: x[1])
    return sorted_by_second

probs = predict_file()
#print(probs)
for prob in probs:
    print(prob)
