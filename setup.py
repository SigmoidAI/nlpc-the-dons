from Model import Models
import DataCleaning
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

stop_words = stopwords.words('english')
porter = PorterStemmer()

X = "1010 ' , 'would ' , 'buy ' , 'love ' , 'game ' , 'like ' , 'can ' , 'hi ' , 'love ' , 'game ' "
X = DataCleaning.WordExtractor(stop_words=stop_words).transform(X)
X = DataCleaning.ApplyStemmer(stemmer=porter).transform(X)

Models.predict(X)