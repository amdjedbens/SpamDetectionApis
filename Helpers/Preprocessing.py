from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
nltk.download('stopwords')
import re



import numpy as np


text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"
def preprocess(text, stem=False):
    stemmer = PorterStemmer()
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


stop_words = stopwords.words('english')
def PreProcess(X,Y=None):
    X = X.apply(lambda x: preprocess(x,True))


    #Wor2vec part/ tokenization
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(X)


    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1000
    # print("Vocabulary Size :", vocab_size)

    X.head().reset_index()
    try:
        Y.head().reset_index()
    except:
        pass
    X = pad_sequences(tokenizer.texts_to_sequences(X),maxlen = 50)
    return X,Y,vocab_size,word_index