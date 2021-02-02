from tweet_sent_predictor.tools.extract import create_df, save_df


df = create_df("tweet_sent_predictor/data/train.txt")
save_df(df,"tweet_sent_predictor/data/train_proper.csv")