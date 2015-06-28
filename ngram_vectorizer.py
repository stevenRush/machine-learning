import numpy as np
import pandas as pd
from nltk.util import ngrams
from collections import Counter
from itertools import chain

def ngram_vectorize(sentence, n):
    all_ngrams = map(lambda x: ''.join(x), product(string.ascii_lowercase, repeat=n))
    grams = chain.from_iterable(ngrams(word, n) for word in sentence.split())
    grams = map(lambda x: ''.join(x), grams)
    grams_counter = Counter(grams)
    return np.array([grams_counter[gramm] for gramm in all_ngrams])

def get_ngrams(sentences, n):
    return np.vstack(ngram_vectorize(sentence, n) for sentence in sentences)

def get_ngrams_listn(sentences, list_n):
    return np.hstack(get_ngrams(sentences, n) for n in list_n)