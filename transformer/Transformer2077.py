from sklearn.base import TransformerMixin
import pandas as pd
class Transformer2077(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.apply(str.lower)
        X = X.str.split(r"[, \-!?:]+")
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

#tweets = pd.read_csv("../train_proper.csv")
#transformer = Transformer2077()
#print(transformer.transform(tweets["body"]).head(5))
