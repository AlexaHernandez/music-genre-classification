import re

from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class CustomPreprocessor(TransformerMixin, BaseEstimator):
    """Custom Transformer to handle all preprocessing steps, including preliminary and experimented steps."""

    def __init__(self, rm_stopwords=False, stem=False, lemmatize=False):
        self.rm_stopwords = rm_stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    def transform(self, X):
        """Return preprocessed songs."""
        df = pd.DataFrame(data=np.copy(X), columns=["lyrics"])
        df["lyrics"] = df.lyrics.apply(lambda x: self._preprocess(x))
        return df.lyrics.values

    def fit(self, X, y=None):
        return self

    def _preprocess(self, song):
        song = self._rm_punc_lower_tokenize(song)
        if self.rm_stopwords:
            song = self._rm_stopwords(song)
        if self.stem:
            song = self._stem(song)
        if self.lemmatize:
            song = self._lemmatize(song)
        return song


    def _rm_stopwords(self, song):
        """Remove stopwords from song."""
        stopwords_eng = stopwords.words('english') + ["nt"]  # Add 'nt' since we tokenized reviews before hand
        return [word for word in song if word not in stopwords_eng]

    def _rm_punc_lower_tokenize(self, song):
        """Remove all non-alphabet characters from the song, convert the song to lowercase, and tokenize it."""
        song = re.sub(r'[^a-zA-Z ]', ' ', song)
        return word_tokenize(song.lower())

    def _stem(self, song):
        """Stem the song."""
        ps = PorterStemmer()
        return [ps.stem(word) for word in song]


    def _lemmatize(self, song):
        """Lemmatize the song."""
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(word) for word in song]
