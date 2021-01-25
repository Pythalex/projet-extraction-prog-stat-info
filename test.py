from sklearn.pipeline import Pipeline
from transformer.LowerCaseTransformer import LowerCaseTransformer
from transformer.MentionFlagger import MentionFlagger
from transformer.NumberFlagger import NumberFlagger
from transformer.SplitterPunctuation import SplitterPunctuation
from transformer.URLFlagger import URLFlagger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("train_proper.csv")

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