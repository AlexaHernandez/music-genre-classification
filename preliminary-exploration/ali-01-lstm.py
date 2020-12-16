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




from tensorflow import keras
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
vectorizer = CountVectorizer(binary=True)
X = []
y = []


df = pd.read_csv('scraped-lyrics-v2.csv')

X = df['lyrics'].fillna(' ')
X = np.array(X)
y = df['genres'].fillna(' ')
y = np.array(y)

X = X.reshape((1, 1, 79877))
y = y.reshape((1, 1, 79877))

x = [i for i in range(65223, 72312)]
x = np.array(x).reshape((1, 1, 7089))
y = [i for i in range(65223, 72312)]
y = np.array(y).reshape((1, 1, 7089))

x_test = [i for i in range(65223, 72312)]
x_test = np.array(x_test).reshape((1, 1, 7089))
y_test = [i for i in range(65223, 72312)]
y_test = np.array(y_test).reshape((1, 1, 7089))


#LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(1, 7089), return_sequences=True))
model.add(Dense(7089))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, validation_data=(x_test, y_test))
y_predict = model.predict(x_test)


