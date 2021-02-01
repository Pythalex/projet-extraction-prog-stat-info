from sklearn.base import TransformerMixin
import re

class NumberFlagger(TransformerMixin):

    def __init__(self, flag="<NUMBER>"):
        self.regex = re.compile("[0-9]+")
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
        
        shift = 0
        for match in self.regex.finditer(sentence):
            f, t = match.span()
            replaced = match.group()
            # replace matched sequence with flag
            sentence = sentence[:f+shift] + self.flag + sentence[t+shift:]

            shift += len(self.flag) - len(replaced)

        return sentence


if __name__ == "__main__":

    import pandas as pd

    tweets = pd.read_csv("train_proper.csv")
    transformer = NumberFlagger()
    
    test = tweets.iloc[[0, 691, 100, 101]]
    tsf = transformer.transform(test["body"])
    for i in range(test.shape[0]):
        print(test.iloc[i]["body"])
        print(tsf.iloc[i])
        print()