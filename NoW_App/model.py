# Saving model
# import libraries used to develop the model

# Import pandas for data handling
import pandas as pd

# Import our text vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our classifiers
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our metrics to evaluate our model
import os
import pickle #saves our trained model to the disk 
import requests
import json

df_concat = pd.read_csv("D:\\Github\\news-of-the-world\\data\\merged_df.csv")

X = df_concat['cleaned_text'].values

y = df_concat['isfake'].values

vectorizer = TfidfVectorizer()

vectorizer.fit(X)

X = vectorizer.transform(X)

model = MultinomialNB(alpha = .05)

model.fit(X,y)

pickle.dump(vectorizer, open('models/vectorizer.pkl','wb'))

pickle.dump(model, open('models/naivebayes.pkl','wb'))
