from sklearn.pipeline import Pipeline
from tweet_sent_predictor.transformer.LowerCaseTransformer import LowerCaseTransformer
from tweet_sent_predictor.transformer.MentionFlagger import MentionFlagger
from tweet_sent_predictor.transformer.NumberFlagger import NumberFlagger
from tweet_sent_predictor.transformer.SplitterPunctuation import SplitterPunctuation
from tweet_sent_predictor.transformer.URLFlagger import URLFlagger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("tweet_sent_predictor/data/train_proper.csv")

pipe = Pipeline([
    ("lower case", LowerCaseTransformer()),
    ("URL flag", URLFlagger()),
    ("Mention flag", MentionFlagger()),
    ("Number flag", NumberFlagger()),
    ("Tokenize", SplitterPunctuation()),
    ("Count vector", CountVectorizer(analyzer=lambda x : x))
])

df_transformed = pipe.fit_transform(df["body"], df["opinion"])

print(df.head())
print()
print(df_transformed)