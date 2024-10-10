import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

corpus = [
    "I love dogs",
    "I love cats",
    "I love programming"
]

# tokenize the corpus
sentences = [word_tokenize(doc) for doc in corpus]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

vector = model.wv['love']
print(vector)
