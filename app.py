import flask
import os
import pickle
import pandas as pd
import skimage
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def to_lower(document): 
    return document.lower()

def remove_punctuation(document):    
    document = re.sub(r'[^\w\s]','',document)
    return document

def remove_stopword(string):
    words = word_tokenize(string)
    accepted_bag = []
    for element in words:
        if element not in stopwords:
            accepted_bag.append(element)
            
    string = ' '.join(accepted_bag)
    
    return string

def text_pipeline(input_string):
    input_string = to_lower(input_string)
    input_string = remove_punctuation(input_string)
    input_string = remove_stopword(input_string)
    return input_string

app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/naivebayes.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']

        cleaned_input = text_pipeline(user_input_text)

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([cleaned_input])
        
        # Make a prediction 
        x_pred = model.predict(X)
        
        # Get the first and only value of the prediction.
        prediction = x_pred[0]

        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)

        # Get the value of the first, and only, predicted proba.
        predicted_proba = predicted_probas[0]

        # The first element in the predicted probabs is % democrat
        percent_true = predicted_proba[0]

        # The second elemnt in predicted probas is % republican
        percent_fake= predicted_proba[1]

        return flask.render_template('main.html', 
            input_text=user_input_text,
            result=prediction,
            percent_fake=percent_fake,
            percent_true=percent_true)

@app.route('/images/')
def images():
    return flask.render_template('images.html')
    
@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')

if __name__ == '__main__':
    app.run(debug=True)