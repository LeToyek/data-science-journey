import feature_extraction as fe
import logistic_regression as lr
import pandas as pd
import preprocess

preproc = preprocess.Preprocess()
feature_extract = fe.FeatureExtraction()
model_lr = lr.LogisticRegression(verbose=True)

print("Load data")
df = pd.read_csv('../data/opini_film.csv',sep=',', nrows = 200)
print(df.head())

df['Sentiment'] = df['Sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

print("Preprocessing data")
df['preprocessed'] = df['Text Tweet'].apply(preproc.preprocess)

print("Create frequency matrix")
X, vocab = feature_extract.create_frequency_matrix(df['preprocessed'])
print(X.shape)
y = df['Sentiment'].values
model_lr.fit(X,y)

test_data = {'text': ["film ini sangat bagus", "film sampah jelek bgt tidak suka bintang 1"]}
test_df = pd.DataFrame(test_data)

test_df['preprocessed'] = test_df['text'].apply(preproc.preprocess)
X_test, _ = feature_extract.create_frequency_matrix(test_df['preprocessed'],vocab_list=vocab)

y_pred = model_lr.predict(X_test)
print(y_pred)

