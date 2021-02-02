from sklearn.base import TransformerMixin
from tweet_sent_predictor.transformer.SplitterPunctuation import SplitterPunctuation
import re
#To be applied after tokenization
class StopWordFilter(TransformerMixin):
    
    #Maybe will add some more when transformers are more fleshed out : https://note.nkmk.me/en/python-long-string/

    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
        'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
        'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'd',
        's', 't', 'can', 'will', 'just', 'should', "ve", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']

    #stop_words=['i', 'me', 'a', 'is', 'for', 'of', 'on', 'the', 'to', 'her', 'we', 'so', 'how', 'in', 'who', 'what']

    def __init__(self, stop_words=stop_words):
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    # def transform(self, X):
    #     X = X.copy()
    #     X = X.apply(lambda x : [el for el in x if el not in self.stop_words])
    #     return X

    def transform(self, X):
        X = X.copy()
        X = X.apply(self.replace)
        return X
    
    def replace(self, row) :
        for stop_word in self.stop_words :
            row = re.sub(r'\b'+stop_word+r'\b',"",row)
        return row

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    


if __name__ == "__main__":

    import pandas as pd

    tweets = pd.read_csv("train_proper.csv")
    transformer = StopWordFilter()
    # splitter = SplitterPunctuation()
    # tweets_split = splitter.transform(tweets["body"])

    # test = tweets_split.iloc[list(range(5))]
    test = tweets.iloc[list(range(5))]
    tsf = transformer.transform(test["body"])
    for i in range(test.shape[0]):
        #print(test.iloc[i])
        print(tsf.iloc[i])
        print()
