import nltk
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
import csv

#Need to:
#Stem, Lemmatize, remove infrequent words and stopwords from each corpus
#Change words to unigram representation feature vectors using NLTK
#Experiment with regularization/smoothing on vectors using scikit-learn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from pprint import pprint
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.datasets import make_blobs


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
vectorizer = CountVectorizer(binary=True)
X = []
y = []

df = pd.read_csv('scraped-lyrics-v1.csv')
X = df['lyrics'].fillna(' ')
y = df['genre'].fillna(' ')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vector = TfidfVectorizer()
X_train_vector = vector.fit_transform(X_train)

X_train_balanced = X_train_vector
y_train_balanced = y_train
X_test_vector = vector.transform(X_test)


#Naive Bayes
nb = MultinomialNB()
parameters_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
grid_search_nb = GridSearchCV(estimator=nb, param_grid=parameters_nb, verbose=2, scoring='accuracy')

nb.fit(X_train_balanced, y_train_balanced)
y_predict = nb.predict(X_test_vector)

print("Naive Bayes")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_predict) * 100))
