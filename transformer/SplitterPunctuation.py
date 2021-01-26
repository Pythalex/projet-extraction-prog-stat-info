from sklearn.base import TransformerMixin


class SplitterPunctuation(TransformerMixin):

    def __init__(self, ponct=r"[, \-!?:\"]+"):
        self.ponct = ponct

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.str.split(self.ponct)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == "__main__":

    import pandas as pd

    tweets = pd.read_csv("train_proper.csv")
    transformer = SplitterPunctuation()

    test = tweets.iloc[list(range(5))]
    tsf = transformer.transform(test["body"])
    for i in range(test.shape[0]):
        print(test.iloc[i]["body"])
        print(tsf.iloc[i])
        print()