from sklearn.base import TransformerMixin
import re

class HashtagToWords(TransformerMixin):

    def __init__(self):
        self.regex = re.compile("#(\w)+")

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
            
            hashtag = match.group()[1:] # remove #
            f, t = match.span()
            splitted = []
            oldsplit = 0
            for wordsmatch in re.finditer("[A-Z]|[0-9]", hashtag):

                split_index = wordsmatch.span()[0]

                splitted.append(hashtag[oldsplit:split_index])

                oldsplit = split_index

            splitted.append(hashtag[oldsplit:])

            splitted = " ".join(splitted).strip()
            sentence = sentence[:f+shift] + splitted + sentence[t+shift:]

            shift += len(splitted) - (len(hashtag)+1)

        return sentence


if __name__ == "__main__":

    import pandas as pd

    tweets = pd.read_csv("tweet_sent_predictor/data/train_proper.csv")
    transformer = HashtagToWords()

    test = pd.Series(["#HelloWorldTwitter bitconneeeeeeeeect yeeha", "#Single test", "#tteeeeee ffffefe saklyut"])
    for i in range(test.shape[0]):
        print(test[i])
        print(transformer.transform(test[[i]])[i])
        print()

    test = tweets.iloc[[0, 6, 41, 106]]
    tsf = transformer.transform(test["body"])
    for i in range(test.shape[0]):
        print(test.iloc[i]["body"])
        print(tsf.iloc[i])
        print()