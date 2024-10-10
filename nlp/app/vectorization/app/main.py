from keras.datasets import imdb
from sklearn.linear_model import LogisticRegression
import preprocess as pr

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
preproc = pr.Preprocessing()
preproc.preprocess_text
