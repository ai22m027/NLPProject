import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import os.path
from os import path

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('punkt')
#nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from fastai.text.all import *
import fastai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import emoji
import re
import string
from collections import Counter
from stop_words import get_stop_words

class TweetClass():
    
    def __init__(self, path: str) -> None:
        self.raw_data = pd.DataFrame()
        self.clean_data_light = pd.DataFrame()
        self.clean_data_strong = pd.DataFrame()
        self.read_data(path)
        
        #Stemmer and Lemmer
        nltk.LancasterStemmer
        self.ls = LancasterStemmer()
        self.lem = WordNetLemmatizer()
        
    def read_data(self,path: str) -> None:
        self.raw_data = pd.read_csv(path)
        
        pass
        
    def raw_df(self) -> pd.DataFrame():
        return self.raw_data
    
    @staticmethod
    def _additional_space_removal(text: str) -> str:
        text = text.strip()
        text = text.split()
        return " ".join(text)
    
    def _stemming_lemming(self, text: str) -> str:
        words = word_tokenize(text)
        #Stemm
        words_stem = [self.ls.stem(w) for w in words]
        #Lemmatize
        words_lem = [self.lem.lemmatize(w) for w in words_stem]
        return words_lem
    
    @staticmethod
    def _clear_emoji(text: str) -> str:
        for emot in emoji.UNICODE_EMOJI:
            text = re.sub(r'('+emot+')', "_".join(emoji.UNICODE_EMOJI[emot].replace(",","").replace(":","").split()), text)
        return text
    
    @staticmethod  
    def _clear_text_full(text: str) -> str:
        """Lower case and remove quotes, links, html, line brackets, numbers and apostrophes.

        Args:
            text (str): raw text

        Returns:
            str: processed text
        """
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\.com/\S+', '', text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('\'','', text)
        
        return text
    
    @staticmethod  
    def _clear_text_light(text: str) -> str:
        """Remove links and HTML content.

        Args:
            text (str): raw text

        Returns:
            str: processed text
        """
        text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\.com/\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        
        return text
    
    @staticmethod
    def _remove_stopwords(text: str) -> str:
        stop_words = get_stop_words("en")
        stopwords_dict = Counter(stop_words)
        text = ' '.join([word for word in text.split() if word not in stopwords_dict])
        return text
    
    @staticmethod
    def _tokenize(text: str) -> str:
        words = word_tokenize(text)
        return words
    
    @staticmethod
    def _clean_regex(df: pd.DataFrame) -> pd.DataFrame:
        df['content'] = df['content'].map(lambda x: re.sub(r'\W+', ' ', x))
        df['content'] = df['content'].replace(r'\W+', ' ', regex=True)
        return df
    
    def _light_clean(self) -> pd.DataFrame:
        data = self.raw_data.copy()
        try:
            data = data.drop(['link','retweets','favorites','mentions','hashtags'], axis=1)
        except: # col do not exist
            pass
        
        data["content"] = data["content"].apply(lambda x: self._clear_text_light(x))
        
        #drop empty tweets
        data = data[data['content'].str.len() > 0]
        data["content"] = data["content"].apply(lambda x: self._tokenize(x))
        
        return data
    
    def _strong_clean(self) -> pd.DataFrame:
        data = self.raw_data.copy()
        try:
            data = data.drop(['link','retweets','favorites','mentions','hashtags'], axis=1)
        except: # col do not exist
            pass
        
        data["content"] = data["content"].apply(lambda x: self._clear_text_full(x))
        data = self._clean_regex(data)
        data["content"] = data["content"].apply(lambda x: self._remove_stopwords(x))
        data["content"] = data["content"].apply(lambda x: self._stemming_lemming(x))
        
        #drop empty tweets
        data = data[data['content'].str.len() > 0]
        
        return data
    
    def clean_data(self) -> None:
        self.clean_data_light = self._light_clean()
        self.clean_data_strong = self._strong_clean()
    
    def visualize_tweet_content(self):
        self.content_length_histogram(self.raw_data, "Raw Tweets")
        self.content_length_histogram(self.clean_data_light, "Light Cleaned Tweets")
        self.content_length_histogram(self.clean_data_strong, "Strong Cleaned Tweets")
        
        #self.most_common_words_vizualization(self.raw_data, "Raw Tweets")
        self.most_common_words_vizualization(self.clean_data_light, "Light Cleaned Tweets")
        self.most_common_words_vizualization(self.clean_data_strong, "Strong Cleaned Tweets")

    # plot functions
    @staticmethod
    def content_length_histogram(df: pd.DataFrame, title:str = "") -> None:
        content_lengths = df['content'].str.len()

        fig, ax = plt.subplots()
        ax.hist(content_lengths, bins=50)

        ax.set_title('Distribution of content length')
        ax.set_xlabel('Content length')
        ax.set_ylabel('Frequency')
        ax.set_title(title)

        plt.show(block = False)
        
    @staticmethod
    def most_common_words_vizualization(df: pd.DataFrame, title:str = "")->None:
        top = Counter([item for sublist in df['content'] for item in sublist])
        df_reduced = pd.DataFrame(top.most_common(20))
        df_reduced.columns = ['Common_words','count']
        df_reduced.style.background_gradient(cmap='Blues')
        
        fig = px.bar(df_reduced, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
        fig.show()

class TweetGeneratorClass():
    
    def __init__(self) -> None:
        self.train_data = fastai.data.core.DataLoaders()
        self.all_tweets = pd.DataFrame()
        self.created_data = pd.DataFrame()
        self.tweet_column_name = str

    def add_data(self, tweets: pd.DataFrame(), tweet_column_name: str, bs: int=64, seq_len: int=72, n_workers: int=4, save_to_csv=True) -> fastai.text.all.DataBlock:
        """Add data to the class

        Args:
            tweets (pd.DataFrame()): Pandas dataframe
            tweet_column_name (str): Column name in the dataframe, which contains the tweets
            bs: (int): batch size
            seq_len: (int): sequence length of sentences to learn from
            n_workers: (int): number of processes
            save_to_csv=True
 
        Returns:
            train_data (fastai.text.all.DataBlock): Train Data in fastai datablock format
            self.all_tweets (pd.DataFrame()): Real Tweets cleaned including real label (=1) in Pandas Dataframe Format
        """
        
        self.tweet_column_name = tweet_column_name
        
        self.all_tweets[self.tweet_column_name] = tweets[self.tweet_column_name]
        if isinstance(self.all_tweets[self.tweet_column_name][0], list):
            self.all_tweets[self.tweet_column_name] = [' '.join(sent) for sent in self.all_tweets[self.tweet_column_name]]
        
        self.all_tweets["label"] = 1 #label (=1 --> real)

        if save_to_csv:
            self.all_tweets[[self.tweet_column_name, "label"]].to_csv("real_Trump_tweets.csv")
            
        dblock = DataBlock(
           blocks=TextBlock.from_df(self.tweet_column_name, seq_len=seq_len, is_lm=True),
           get_x=ColReader('text'))
        self.train_data = dblock.dataloaders(self.all_tweets, bs=bs, n_workers=4)
        #self.train_data.show_batch(max_n=3)
        
        return self.train_data, self.all_tweets

    def create_model(self, data:fastai.text.all.DataBlock):
        """Create Model for Training

        Args:
            data (data:fastai.text.all.DataBlock): fastai datablock

        Returns:
            None
        """
        # Create deep learning model
        self.learner = language_model_learner(data, AWD_LSTM, path = 'Model/', model_dir='tweet_model/')

    def train_model(self, find_lr: bool=True, lr_iters: int=100, const_lr=None, epochs: int=10):
        """Train model

        Args:
            find_lr (bool): Wanna find a godd lr first
            lr_iters (int): Number of iterations to find best learning rate
            const_lr (float): if None, lr finder is used, else use this lr in combination with set epochs for training
            epochs (int): Number of training loops (epochs)

        Returns:
            None
        """
        
        # Use Learning Rate finder algorithm
        if find_lr and const_lr == None:
            # find the appropriate learning rate
            lr = self.learner.lr_find(num_it=lr_iters)
            print("optimal learning rate: ", lr)
            # find the point where the slope is steepest
            self.learner.recorder.plot_lr_find()

            # Fit the model based on selected learning rate
            self.learner.fit_one_cycle(1, 1e-2)
            
        # OR Use standard learning
        else:
            self.learner.fit(epochs, lr=const_lr)
            self.learner.recorder.plot_loss()
            
        # save model to ./Model/tweet_model/
        self.learner.save("trump_tweeter")
    
    def create_tweet(self, begin_sentence: str="random",  sentence_length: int=20) -> str:
        """Generate One Tweet

        Args:
            begin_sentence (str): string each sentence needs to begin with, if 'random': random first 4 words of random original tweet is used.
            sentence_length (int): length of generated sentence, length=number of tokens, not necessarily words only.

        Returns:
            tweet_txt (str): One Tweet
        """
               
        # load model from ./Model/tweet_model/
        self.learner.load("trump_tweeter")

        # Predict Tweets starting from the given words 
        if begin_sentence == "random":
            rand_tweet = self.all_tweets[self.tweet_column_name ][np.random.randint(low=0, high=self.all_tweets.shape[0])]
            if isinstance(rand_tweet, list):
                rand_begin_sentence = " ".join(rand_tweet[0:4])
            else:
                rand_begin_sentence = " ".join(rand_tweet.split(" ")[0:4]) 
            print(rand_begin_sentence)
            tweet_txt = self.learner.predict(rand_begin_sentence, sentence_length, no_unk=True, temperature=0.75)
        else:
            tweet_txt = self.learner.predict(begin_sentence, sentence_length, temperature=0.75)
        return tweet_txt
    
    def create_tweet_set(self, num_of_tweets:int = 100, begin_sentence: str="random", sentence_length: int=20, save_to_csv=True) -> pd.DataFrame():
        """Generate One Tweet

        Args:
            begin_sentence (str): string each sentence needs to begin with, if 'random': random first 4 words of random original tweet is used.
            sentence_length (int): length of generated sentence, length=number of tokens, not necessarily words only.

        Returns:
            self.created_data (pd.DataFrame()): Pandas Dataframe of as many tweets as generated, with 2 columns: "self.tweet_column_name", label (=0 --> fake)
        """
          
        tweet_set = []
        for idx in range(num_of_tweets):
            if begin_sentence == "random":
                rand_tweet = self.all_tweets[self.tweet_column_name ][np.random.randint(low=0, high=self.all_tweets.shape[0])]
                if isinstance(rand_tweet, list):
                    rand_begin_sentence = " ".join(rand_tweet[0:4])
                else:
                    rand_begin_sentence = " ".join(rand_tweet.split(" ")[0:4]) 
                tweet_set.append(self.learner.predict(rand_begin_sentence, sentence_length, no_unk=True, temperature=0.75))
            else:
                tweet_set.append(self.learner.predict(begin_sentence, sentence_length, temperature=1.0))
                
        self.created_data = pd.DataFrame(tweet_set, columns = [self.tweet_column_name])
        self.created_data["label"] = 0
        
        if save_to_csv:
            self.created_data.to_csv("fake_Trump_tweets.csv")
        
        return self.created_data


def main():
    root_path = os.getcwd()
    data_path = f"{root_path}/Data/NLP"

    tweets_csv = "realdonaldtrump.csv"


    data_path = f"{data_path}/{tweets_csv}"
    tweet_col = "content"
    trump_tweets = TweetClass(data_path)
    trump_tweets.clean_data()
    
    trump_tweets.content_length_histogram(trump_tweets.clean_data_light,"Light Clean")
    trump_tweets.most_common_words_vizualization(trump_tweets.clean_data_light,"Light Clean")
    
    trump_tweets.content_length_histogram(trump_tweets.clean_data_strong,"Strong Clean")
    trump_tweets.most_common_words_vizualization(trump_tweets.clean_data_strong,"Strong Clean")

if __name__ == "__main__":
    main()
