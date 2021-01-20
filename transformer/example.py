from sklearn.base import TransformerMixin

class Example(TransformerMixin):

    def __init__(self, mon_param):
        self.mon_param = mon_param

    def fit(X, y=None):
        function(mon_param)
        ...
        return self

    def transform(X):
        X = X.copy()
        ...
        return X

    def fit_transform(X, y=None):
        return self.fit(X, y).transform(X)