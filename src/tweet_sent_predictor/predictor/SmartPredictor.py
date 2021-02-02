from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tweet_sent_predictor.transformer.LowerCaseTransformer import LowerCaseTransformer
from tweet_sent_predictor.transformer.MentionFlagger import MentionFlagger
from tweet_sent_predictor.transformer.NumberFlagger import NumberFlagger
from tweet_sent_predictor.transformer.SplitterPunctuation import SplitterPunctuation
from tweet_sent_predictor.transformer.URLFlagger import URLFlagger
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from tweet_sent_predictor.predictor.LanguagePredictor import LanguagePredictor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tweet_sent_predictor.transformer.MentionFilter import MentionFilter
from tweet_sent_predictor.transformer.HashtagFilter import HashtagFilter
from tweet_sent_predictor.transformer.URLFilter import URLFilter


class SmartPredictor(BaseEstimator, ClassifierMixin):

    pre_lang_detect_pipe = Pipeline([
            ("remove mention", MentionFilter()),
            ("remove hash", HashtagFilter()),
            ("remove url", URLFilter()),
        ])
    language_pred = LanguagePredictor(nbpass=3)
    
    def __init__(self, pipe, pre_lang_detect_pipe=pre_lang_detect_pipe):
        self.pre_lang_detect_pipe = pre_lang_detect_pipe
        self.pipe = pipe
        
    def fit(self, X, y):
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Use the first pipeline to predict languages
        # We don't use a validation set because the lang detector doesn't fit on the data, it will predict the same whether it's new or old data.
        X_lang = self.pre_lang_detect_pipe.fit_transform(X.copy(), y)
        langs = self.language_pred.predict(X_lang)
        english_tweets = np.where(langs == "en")[0]
        
        # Train the second pipeline
        # X is a pandas series so it needs indices to be consistent (first row != 0) while y is a numpy array such that first row is index 0
        self.pipe.fit(X[langs.index[english_tweets]], y[english_tweets])
        
        print("fit")
        
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        
        X_lang = self.pre_lang_detect_pipe.fit_transform(X.copy())
        langs = self.language_pred.predict(X_lang)
        english_tweets = np.where(langs == "en")[0]
        foreign_tweets = np.where(langs != "en")[0]

        foreign_pred = pd.DataFrame({"target" : ("irr")}, index=langs.index[foreign_tweets])
        
        preds = self.pipe.predict(X[langs.index[english_tweets]])
        english_pred = pd.DataFrame({"target": preds}, index=langs.index[english_tweets])

        return pd.concat((foreign_pred, english_pred), axis=0).sort_index()["target"]

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    #def score(self, X, y):
        #"""Mean accuracy score on X,y"""
        #y_pred = self.predict(X)
        
        #diff = [1 if y[i] == y_pred[i] else 0 for i in range(len(y))]
        #return diff.sum() / len(diff) 

        
