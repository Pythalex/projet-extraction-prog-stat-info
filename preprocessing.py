import preprocessor as p #forming a separate feature for cleaned tweets
import pandas as pd

tweets = pd.read_csv("train_proper.csv")
tweets = tweets.convert_dtypes("str")

#print(tweets.dtypes)
#print(tweets.head(5))

def clean_body(tweet) : 


def produce_df_features(df) :
    for i,v in enumerate(df['body']):
        df.loc[i,'clean'] = p.clean(v)
    print(df.head(5))
    df.to_csv("test.csv",index=False)

produce_df_features(tweets)
