from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtraction:
    def __init__(self):
        pass

    def word2vec(self, corpus):
        model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def count_vectorizer(self, corpus):
        vectorizer = CountVectorizer(max_features=5000)
        X = vectorizer.fit_transform(corpus)
        return X.toarray()

    def tfidf_vectorizer(self, corpus):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(corpus)
        return X.toarray()