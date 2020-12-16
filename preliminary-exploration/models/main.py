import pickle
from os.path import dirname, abspath, join

import pandas as pd
import numpy as np

from preprocessor import CustomPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, "data", "scraped-lyrics-v1.csv")
RESULTS = join(ROOT, "results")


def load_data(f):
    """Return pandas Dataframe containing data loaded from CSV f."""
    return pd.read_csv(f)


def identity(x: str):
    return x


def build_pipe(clf):
    """Return Pipeline with clf as the final estimator."""
    return Pipeline([('pre', CustomPreprocessor()),  # Preprocessing step
                     ('vect', TfidfVectorizer(  # Feature extraction step
                         tokenizer=identity,  # Since corpus is already tokenized
                         max_df=0.70,
                         preprocessor=None,  # Since preprocessing is done by CustomPreprocessor
                         lowercase=False)),
                     ('clf', clf)])  # Classifier


def build_grid_search_cv(pipe, params):
    """Return GridSearchCV object to search pipe over values in params."""
    return GridSearchCV(estimator=pipe,
                        param_grid=params,
                        scoring='accuracy',
                        cv=3,  # 3-fold cross val
                        verbose=3,
                        return_train_score=True)


def grid_search_nb():
    """Return GridSearchCV object to tune a multinomial naive Bayes classifier."""
    pipe = build_pipe(MultinomialNB())  # Build Pipeline with Multinomial Naive Bayes clf
    grid_params = [{'pre__lemmatize': [True, False],  # Set hyperparameter values to search
                    'pre__rm_stopwords': [True, False],
                    'clf__alpha': [0.1, 1, 5]}]
    return build_grid_search_cv(pipe, grid_params)  # Return GridSearchCV for given pipe and grid_params


def run_grid_search(gs, X_train: np.ndarray, y_train: np.ndarray, name: str):
    """Run grid search on GridSearchCV object gs with X_train and y_train.
       Save CSV file of results and pickles the GridSearchCV object gs."""
    gs.fit(X_train, y_train)
    df_scores = pd.DataFrame(gs.cv_results_)
    df_scores.to_csv(join(RESULTS, "gs_results_{name}.csv".format(name=name)))
    with open(join(RESULTS, "gs_{name}.pkl".format(name=name)), "wb+") as f:
        pickle.dump(gs, f)


def lower(X):
    return lambda x: x.lower()


if __name__ == "__main__":
    # Load data
    df = load_data(DATA)
    df = df[df.lyrics.str.lower() != '[instrumental]']  # Remove instrumental songs
    df = df[:25000]
    X = df.lyrics.values
    y = df.genre.values

    # Partition data into train/test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # Train and tune a Multinomial NB classifier
    gs = grid_search_nb()
    run_grid_search(gs, X_train, y_train, 'nb')
