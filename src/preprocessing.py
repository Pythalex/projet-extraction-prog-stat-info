from tweet_sent_predictor.tools.preprocessing import produce_df_features
import pandas as pd


tweets = pd.read_csv("tweet_sent_predictor/data/train_proper.csv")
tweets = tweets.convert_dtypes("str")
produce_df_features(tweets)