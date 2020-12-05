import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    
    ''' 
    this functions loads the data from a the sql db file
    args: 
        database_filepath: location of the db file
    Returns:
    X: the message column
    y: the categories
    y_cols: the names of the categories
    '''    
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='Disaster_Response_Database', con=engine)
    y_cols=[]
    for i in df.columns:
        if i not in ['id','message','original','genre']:
            y_cols.append(i)
    df=df.dropna(subset=y_cols)
    X = df['message']

    y=df[y_cols]
    
    return X,y,y_cols


def tokenize(text):
    """
    This function tokenize the gieven text.
    args:
        text: it is the text body to tokenized.
    Return:
        tokens: tokenized version of the text body.
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove the stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    
    
    return tokens


def build_model():
    '''
    Building a machine learning pipeline which takes in the message column as input, and 
    it classifies the results in one of the 36 categories in the dataset. 
    args:
        None
    Returns:
        cv2: the machine learning model
    '''
           
    pipeline2 = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters2={'features__text_pipeline__tfidf__use_idf': (True, False)}
    cv2 = GridSearchCV(pipeline2, param_grid=parameters2,n_jobs=-1)
    
    return cv2


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    It display the precision, recall and f1 score for each of the 36 categories.
    args:
        model: the model to be evaluated
        X_test: test dataset
        y_test: categories for each message of X_test
        category_names: list of categories for the messages to be classified
    
    Returns: None

    """
    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names) ) 
    pass
    
    
    
def save_model(model, model_filepath):
    """
    It save best parameters of the machine learning pipeline model as pickle file.
    args:
        model: machine learning pipeline model
        model_filepath: file path for saving the model
    Returns: None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()