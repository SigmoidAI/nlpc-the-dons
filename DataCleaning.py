#Importing the libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import os
from os.path import join
import re
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from functools import reduce
from operator import add
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('stopwords')
nltk.download('punkt')

regex = r'[^a-zA-Z]'

class WordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words = None):
        self.__stop_words = stop_words
    def fit(self, X, y=None):

        """ This function is responsible for finding the hapaxes."""

        self.__hapaxes = []
        
        fdist = FreqDist()
        
        self.__hapaxes = fdist.hapaxes()
        return self

    def transform(self, X, y=None):

        """This function is deleting the stopwords, resize the text to lower case and remove all the numbers from the text."""

        X = re.sub(regex, " ", X)
        X = X.lower()
        self.fit(X)
        self.common_hapaxes = list(reduce(add, [self.__hapaxes]))
        X = ' '.join([word for word in word_tokenize(X)
                                        if word not in self.common_hapaxes])
        
        if self.__stop_words is not None:
            X = ' '.join([word for word in word_tokenize(X)
                            if word not in self.__stop_words])
        return X

class ApplyStemmer(BaseEstimator, TransformerMixin):
    def __init__(self, stemmer = None):
        self.stemmer = stemmer
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):

        """This function amply the stemmer on every word in the sentence."""

        X =  ' '.join([self.stemmer.stem(word) for word in word_tokenize(X)])
        vectorizer=CountVectorizer()
        X = vectorizer.fit_transform([X])

        X = X.toarray()
        
        return X
