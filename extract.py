import pandas as pd


def create_df(path) :
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    l = []
    for line in lines :
        tweet = line.split(" ",1)
        body = tweet[1]
        meta_data = tweet[0][1:-1].split(",")
        opinion = meta_data[1]
        brand = meta_data[2]
        l.append([opinion,brand,body])
    df = pd.DataFrame(l, columns =['opinion', 'brand', 'body'])
    return df



def save_df(df, path) :
    df.to_csv(path,index=False)

df = create_df("train.txt")
save_df(df,"train_proper.csv")
