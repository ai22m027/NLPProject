import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import emoji
import re
import string

import nltk
nltk.download('punkit')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

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
        data = data.drop(['link','retweets','favorites','mentions','hashtags'], axis=1)
        
        data["content"] = data["content"].apply(lambda x: self._clear_text_light(x))
        
        #drop empty tweets
        data = data[data['content'].str.len() > 0]
        data["content"] = data["content"].apply(lambda x: self._tokenize(x))
        
        return data
    
    def _strong_clean(self) -> pd.DataFrame:
        data = self.raw_data.copy()
        data = data.drop(['link','retweets','favorites','mentions','hashtags'], axis=1)
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

    
