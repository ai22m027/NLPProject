import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from TweetClasses.tweetClass import TweetClass

from collections import Counter
from stop_words import get_stop_words

class TweetClass():
    
    def __init__(self, path: str) -> None:
        self.train_data = pd.DataFrame()
        self.created_data = pd.DataFrame()

        
    def add_data(self, tweets: TweetClass) -> None:
        """Add data to the class

        Args:
            df (pd.DataFrame): _description_
        """
        pass
    
    def train_model(self):
        pass
    
    def create_tweet(self) -> str:
        tweet_txt = ""
        return tweet_txt
    
    def create_tweet_class(self) -> None:
        pass
    
    def create_tweet_set(self, num_of_tweets:int = 100) -> None:
        dataframe = pd.DataFrame()
        for idx in range(num_of_tweets):
            dataframe = pd.concat(dataframe, self.create_tweet())
        self.created_data = dataframe

    
