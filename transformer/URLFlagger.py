from sklearn.base import TransformerMixin
import re

class NumberFlagger(TransformerMixin):

    def __init__(self, flag="<URL>"):
        self.regex = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
        self.flag = flag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.apply(self.replace)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def replace(self, sentence):
        sentence = str(sentence) # copy
        
        for match in self.regex.finditer(sentence):
            f, t = match.span()
            # replace matched sequence with flag
            sentence = sentence[:f] + self.flag + sentence[t:]

        return sentence


if __name__ == "__main__":

    import pandas as pd

    tweets = pd.read_csv("../train_proper.csv")
    transformer = NumberFlagger()
    
    for testline in [0, 222, 314, 347]:
        test = tweets.iloc[testline][["body"]]
        print(test["body"])
        print(transformer.transform(test)["body"])
        print()