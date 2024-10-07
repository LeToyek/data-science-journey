import re

import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import \
    StopWordRemoverFactory


class Preprocess:
    def __init__(self):
        self.stop_factory = StopWordRemoverFactory()
        self.more_stopword = ['dengan','ia','bahwa','oleh']
        self.indonesian_stopwords = self.stop_factory.get_stop_words() + self.more_stopword
        self.stopword = self.stop_factory.create_stop_word_remover()
        self.stem_factory = StemmerFactory()
        self.stemmer = self.stem_factory.create_stemmer()

    def _tokenize(self, text) -> list[str]:
        return word_tokenize(text)
      
    def _stemming(self, text) -> str:
        return self.stemmer.stem(text)
    
    def _remove_stopwords(self, text) -> list:
        return [word for word in text if word not in self.indonesian_stopwords]
    
    def preprocess(self, text):
        print(text)
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        tokens = self._tokenize(text)

        tokens = self._remove_stopwords(tokens)

        tokens = [self._stemming(token) for token in tokens]
        print(tokens)
        return tokens
  
    
