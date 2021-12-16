import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('preprocess_tweets.csv', sep=',')
    tweets = df['clean'].values
    mx_len = max(tweets, key=lambda x: len(x.split(" ")))
    print(len(mx_len.split(' ')))
