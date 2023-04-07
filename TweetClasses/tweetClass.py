import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import emoji
import re
import string

import nltk
nltk.download('punkit')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

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
    
    @staticmethod
    def spell_correction(text: str):
        pass
    
    @staticmethod
    def additional_space_removal(text: str) -> str:
        text = text.strip()
        text = text.split()
        return " ".join(text)
    
    def stemming_lemming(self, text: str) -> str:
        words = word_tokenize(text)
        #Stemm
        words_stem = [self.ls.stem(w) for w in words]
        #Lemmatize
        words_lem = [self.lem.lemmatize(w) for w in words_stem]
        return words_lem
    
    @staticmethod
    def clear_emoji(text: str) -> str:
        for emot in emoji.UNICODE_EMOJI:
            text = re.sub(r'('+emot+')', "_".join(emoji.UNICODE_EMOJI[emot].replace(",","").replace(":","").split()), text)
        return text
    
    @staticmethod  
    def clear_text_full(text: str) -> str:
        """Lower case and remove quotes, links, html, line brackets, numbers and apostrophes.

        Args:
            text (str): raw text

        Returns:
            str: processed text
        """
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('\'','', text)
        
        return text
    
    @staticmethod  
    def clear_text_light(text: str) -> str:
        """Remove links and HTML content.

        Args:
            text (str): raw text

        Returns:
            str: processed text
        """
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        
        return text
    
    def light_clean(self, ) -> pd.DataFrame:
        data = self.raw_data.copy()
        return data
    
    def strong_clean(self) -> pd.DataFrame:
        data = self.raw_data.copy()
        return data
    
    def clean_data(self) -> None:
        self.clean_data_light = self.light_clean()
        self.clean_data_strong = self.strong_clean()
        self.clean_data = self.raw_data.copy()
        pass
    
    def read_data(self,path: str) -> None:
        self.raw_data = pd.read_csv(path)
        pass
    
