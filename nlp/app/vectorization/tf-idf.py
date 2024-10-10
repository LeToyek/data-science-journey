# why the autocomplete is not appearing

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love dogs",
    "I love cats",
    "I love programming"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.toarray())

print(vectorizer.get_feature_names_out())