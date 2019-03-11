__author__ = 'teemu kanstren'

#https://api.github.com/repos/ARMmbed/mbed-os/issues?state=closed&page=12&per_page=1000

import json
import glob
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from string import punctuation

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(punctuation))
stop_words.update(["''", "--", "**"])

closed_files = glob.glob("data/*-closed-*")
open_files = glob.glob("data/*-closed-*")

all_text_tokens = []
component_labels = []

total = 0


def preprocess_report(body_o):
	# replace all but alphabetical and numerical characters and some specials such as /\
	body = re.sub('[^A-Za-z0-9 /\\\_+.,:\n]+', '', body_o)
	# replace URL separators with space so the parts of the url become separate words
	body = re.sub('[/\\\]', ' ', body)
	# finally lemmatize all words for the analysis
	lemmatizer = WordNetLemmatizer()
	# text tokens are basis for the features
	text_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(body.lower()) if word not in stop_words]
	return text_tokens


def process_files(files):
	'''
	load issue descriptions from given files. collect the issue "component" field as label, "body" field as the description.
	preprocess the description to turn it into text features to predict the component

	:param files: filenames/paths to process
	:return: nothing
	'''
	global total

	for filename in files:
		with open(filename, encoding="utf-8") as json_data:
			all_issues = json.load(json_data)
			for issue in all_issues:
				labels = issue["labels"]
				for label in labels:
					if label["name"].startswith("component:"):
						name = label["name"]
						body_o = issue["body"]
						text_tokens = preprocess_report(body_o)
						all_text_tokens.append((text_tokens))
						component_labels.append(name)
						total += 1
	print("-------")

process_files(closed_files)
process_files(open_files)

print("total: ", total)

import gensim

from pandas import DataFrame

#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Build the bigram and trigram models
bigram = gensim.models.Phrases(all_text_tokens, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[all_text_tokens], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

#just print something to see it works
print(trigram_mod[bigram_mod[all_text_tokens[4]]])

#transform identified word pairs and triplets into bigrams and trigrams
text_tokens = [trigram_mod[bigram_mod[text]] for text in all_text_tokens]

#build whole documents from text tokens. some algorithms work on documents not tokens.
texts = [" ".join(tokens) for tokens in text_tokens]

df = DataFrame()

df["texts"] = texts
df["text_tokens"] = text_tokens
df["component"] = component_labels

print(df.head(2))


import numpy as np

#how many issues are there in our data for all the target labels, assigned component counts
value_counts = df["component"].value_counts()
#print how many times each component/target label exists in the training data
print(df["component"].value_counts())
#remove all targets for which we have less than 10 training samples.
#K-fold validation with 5 splits requires min 5 to have 1 in each split. This makes it 2, which is still tiny but at least it sorta runs
indices = df["component"].isin(value_counts[value_counts > 9].index)
#this is the weird syntax i never remember, them python tricks. i think it slices the dataframe to remove the items not in "indices" list
df = df.loc[indices, :]

from sklearn.preprocessing import LabelEncoder

# encode class values as integers so they work as targets for the prediction algorithm
encoder = LabelEncoder()
encoded_label = encoder.fit_transform(df.component)

unique, counts = np.unique(encoded_label, return_counts=True)
print(unique) #the set of unique encoded labels
print(counts) #the number of instances for each label


#which textual label/component name matches which integer label
le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(le_name_mapping)

#which integer matches which textual label/component name
le_id_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
print(le_id_mapping)

features = df["texts"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

#transfor all documents into TFIDF vectors.
#TF-IDF vectors are all same length, same word at same index, value is its TFIDF for that word in that document
features_transformed = vectorizer.fit_transform(features)

#the TFIDF feature names is a long list of all unique words found
print(vectorizer.get_feature_names())
feature_names = np.array(vectorizer.get_feature_names())
print(len(feature_names))

def top_tfidf_feats(row, features, top_n=25):
	''' Get top n tfidf values in row and return them with their corresponding feature names.'''
	topn_ids = np.argsort(row)[::-1][:top_n]
	top_feats = [(features[i], row[i]) for i in topn_ids]
	#create a dataframe where first column is the feature and second is its TFIDF score/value
	df = pd.DataFrame(top_feats)
	df.columns = ['feature', 'tfidf']
	return df

def show_most_informative_features(vectorizer, clf, n=20):
	'''
	Get the top TFIDF features, the ones seemingly giving the most information for a given classifier.

	:param vectorizer: TFIDF vectorizer
	:param clf: classifier using the TFIDF scores
	:param n: count of top features to print
	:return:
	'''
	feature_names = vectorizer.get_feature_names()
	coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
	top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5) #5-fold cross validation

#the classifier to use, the parameters are selected based on a set i tried before
clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=1, min_samples_split=5)

results = cross_val_score(clf, features_transformed, encoded_label, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

#show_most_informative_features(vectorizer, clf)

#fit the classifier on the TFIDF transformed word features, train it to predict the component
clf.fit(features_transformed, encoded_label)
probabilities = clf.predict_proba(features_transformed[0])
print(probabilities)

import requests

def predict_component(issue):
	'''
	use this to get a set of predictions for a given issue.

	:param issue: issue id from github.
	:return: list of tuples (component name, probability)
	'''
	#first load text for the given issue from github
	url = "https://api.github.com/repos/ARMmbed/mbed-os/issues/" + str(issue)
	r = requests.get(url)
	url_json = json.loads(r.content)
	print(url_json)
	#process the loaded issue data to format matching what the classifier is trained on
	issue_tokens = preprocess_report(url_json["body"])
	issue_tokens = trigram_mod[bigram_mod[issue_tokens]]
	issue_text = " ".join(issue_tokens)
	features_transformed = vectorizer.transform([issue_text])
	#and predict the probability of each component type
	probabilities = clf.predict_proba(features_transformed)
	result = []
	for idx in range(probabilities.shape[1]):
		name = le_id_mapping[idx]
		prob = (probabilities[0, idx]*100)
		prob_str = "%.2f%%" % prob
		print(name, ":", prob_str)
		result.append((name, prob_str))
	return result




