import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class Preprocessing:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.ps = PorterStemmer()
    
    def _tokenize(self, text) -> list[str]:
        return word_tokenize(text)
    
    def _remove_stopwords(self, text) -> list:
        return [word for word in text if word not in self.stopwords]
    
    def _stemming(self, text) -> list:
        return [self.ps.stem(word) for word in text]
    
    def preprocess_text(self, text):
        text = text.lower()
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._stemming(tokens)
        return tokens