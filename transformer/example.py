class Example(TransformerMixin):

    def fit(X, y=None):
        ...
        return self

    def transform(X):
        X = X.copy()
        ...
        return X

    def fit_transform(X, y=None):
        return self.fit(X, y).transform(X)