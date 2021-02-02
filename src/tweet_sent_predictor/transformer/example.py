from sklearn.base import TransformerMixin

function = lambda x:x

class Example(TransformerMixin):

    def __init__(self, mon_param):
        self.mon_param = mon_param

    def fit(self, X, y=None):
        function(self.mon_param)
        ...
        return self

    def transform(self, X):
        X = X.copy()
        ...
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)