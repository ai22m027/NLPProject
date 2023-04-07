import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import os.path
from os import path

import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from TweetClasses.tweetClass import TweetClass

def main():
    root_path = os.getcwd()
    data_path = f"{root_path}/Data/NLP"
    
    #trumptweets - pre 2018(?)
    #realdonaldtrump - post 2018(?)
    tweets_csv = "realdonaldtrump.csv"
    
    data_path = f"{data_path}/{tweets_csv}"
    
    trump_tweets = TweetClass(data_path)
    trump_tweets.clean_data()
    trump_tweets.visualize_tweet_content()

    pass

if __name__ == "__main__":
    main()