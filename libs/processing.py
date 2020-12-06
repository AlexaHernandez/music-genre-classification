import copy

import nltk
import nltk.stem
import nltk.corpus
for package in ['wordnet', 'averaged_perceptron_tagger', 'punkt', 'stopwords']:
    nltk.download(package, quiet=True, raise_on_error=True)

import sklearn
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_extraction.text
import sklearn.utils
import sklearn.utils.testing
import sklearn.exceptions
import sklearn.multiclass
import sklearn.preprocessing

import pandas as pd
import numpy as np

class Lemmatizer():
    def __init__(self, stop_words):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.tag_prefix_dict = {
            'J': nltk.corpus.wordnet.ADJ,
            'N': nltk.corpus.wordnet.NOUN,
            'V': nltk.corpus.wordnet.VERB,
            'R': nltk.corpus.wordnet.ADV
        }
        self.stop_words = stop_words
    
    def _lemmatize(self, token, pos):
        if token in self.stop_words:
            return token # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
        return self.lemmatizer.lemmatize(token, pos)
    
    def __call__(self, document):        
        return [
            self._lemmatize(token, pos=self.get_tag_class(tag))
            for token, tag in nltk.pos_tag(nltk.word_tokenize(document))
        ]
    
    def get_tag_class(self, tag):
        prefix = tag[0].upper()
        return self.tag_prefix_dict.get(prefix, nltk.corpus.wordnet.NOUN)

class Stemmer():
    def __init__(self, stop_words):
        self.stemmer = nltk.stem.PorterStemmer()
        self.stop_words = stop_words
    
    def _stem(self, token):
        if token in self.stop_words:
            return token # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
        return self.stemmer.stem(token)
    
    def __call__(self, document):        
        return [self._stem(token) for token in nltk.word_tokenize(document)]

class StopWords():
    def __init__(self, description, words=None):
        if description == 'nltk_english':
            words_ = nltk.corpus.stopwords.words('english')
        else:
            assert type(words) == list
            words_ = copy.deepcopy(words)
        
        self.description = description
        self.words = words_
    
    def __str__(self):
        return self.description

def fit_vectorizer(X_data, tokenizer, stop_words, min_df):    
    tokenized_stop_words = nltk.word_tokenize(' '.join(stop_words.words))
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=tokenizer(stop_words.words),
        stop_words=tokenized_stop_words,
        min_df=min_df,
        lowercase=True
    )
    vectorizer.fit_transform(X_data)
    return vectorizer

def get_top_k_predictions(model, x, k=3):
    proba = model.predict_proba(x)
    return (-proba).argsort(axis=1)[:, :k]

def top_k_accuracy_score(truth, top_k_predictions):
    hits = 0
    for i in range(truth.shape[0]):
        true_labels = truth[i].nonzero()[0]
        hits += 1 if np.in1d(top_k_predictions[i], true_labels).any() else 0
    return hits/truth.shape[0]

@sklearn.utils.testing.ignore_warnings(category=sklearn.exceptions.ConvergenceWarning)
def random_search(models, data, search_params, n_datasets=2, n_models=3, multi_label=False):
    results = [{} for i in range(n_datasets)]

    data_set_variations = [
        choose_random_params(search_params['data'])
        for i in range(n_datasets)
    ]
    model_variations = {
        model_name: [choose_random_params(search_params['model'][model_name]) for i in range(n_models)]
        for model_name in search_params['model'].keys()
    }
    
    for i in range(n_datasets):
        print(f'Data set variation {i+1}/{n_datasets}')

        data_params = data_set_variations[i]

        print('\tFitting vectorizer...')
        vectorizer = fit_vectorizer(data['X_train_raw'], **data_params)

        X_train = vectorizer.transform(data['X_train_raw'])
        X_valid = vectorizer.transform(data['X_valid_raw'])
        X_test = vectorizer.transform(data['X_test_raw'])

        for model_name, model_class in models.items():
            for j in range(n_models):
                print(f'\t{model_name} {j+1}/{n_models}')

                model_params = model_variations[model_name][j]

                if multi_label:
                    model = sklearn.multiclass.OneVsRestClassifier(model_class(**model_params))
                    binarizer = sklearn.preprocessing.MultiLabelBinarizer()
                    binarizer.fit(data['y_train'])
                    accuracy_function = top_k_accuracy_score
                else:
                    model = model_class(**model_params)
                    accuracy_function = sklearn.metrics.accuracy_score

                y_train = binarizer.transform(data['y_train']) if multi_label else data['y_train']
                y_valid = binarizer.transform(data['y_valid']) if multi_label else data['y_valid']
                y_test = binarizer.transform(data['y_test']) if multi_label else data['y_test']

                model.fit(X_train, y_train)

                valid_predictions = get_top_k_predictions(model, X_valid, k=2) if multi_label else model.predict(X_valid)
                test_predictions = get_top_k_predictions(model, X_test, k=2) if multi_label else model.predict(X_test)
                
                valid_accuracy = accuracy_function(y_valid, valid_predictions)
                
                # This number is only looked at once at the very end when the best models have been chosen based on validation accuracy
                test_accuracy = accuracy_function(y_test, test_predictions)
                if not multi_label:
                    test_confusion_matrix = sklearn.metrics.confusion_matrix(y_test, test_predictions, labels=sorted(model.classes_), normalize='true')
                    df_test_confusion_matrix = pd.DataFrame(test_confusion_matrix, index=sorted(model.classes_), columns=model.classes_)

                if results[i].get(model_name, None) is None:
                    results[i][model_name] = []

                results[i][model_name].append({
                    'model_params': model_params,
                    'valid_accuracy': valid_accuracy,
                    'test_accuracy': test_accuracy,
                    'df_test_confusion_matrix': df_test_confusion_matrix if not multi_label else None
                })
    
    return data_set_variations, results

def choose_random_params(parameters):
    return {
        name: np.random.choice(values)
        for name, values in parameters.items()
    }
