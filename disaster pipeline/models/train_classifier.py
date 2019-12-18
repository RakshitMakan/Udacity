import sys
import os
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{1}{0}'.format(database_filepath, os.getcwd()))
    print(engine)
    print('sqlite:///{1}/{0}'.format(database_filepath, os.getcwd()))
    df = pd.read_sql_table('message', con=engine)[0:1000]
    print(df.head())
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    print(len(category_names))
    return X, y, category_names

def tokenize(text):
    """tokenize and transform input text. Return cleaned text
    
    parameters:
    text -  the message data from the data base

    return:
    clean_tokens -  The cleaned data
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # take out all punctuation while tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Return Grid Search model with pipeline and Classifier

    parameters:
    
    return:
    cv :  the estimator
    
    """
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv
    
def evaluate_model(model, X_test, y_test, category_names):
    """Print model results
    INPUT
    model -- required, estimator-object
    X_test -- required
    y_test -- required
    category_names = required, list of category strings
    OUTPUT
    None
    """
    # Get results and add them to a dataframe.
    print(y_test.columns)
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = y_test.columns
    print(y_pred.columns)
    print(y_pred.head())
    print(classification_report(y_test, y_pred))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])

def save_model(model, model_filepath):
    """
    Save model as pickle file
    
    parameters:
    model - the final model
    model_filepath : the path of model

    return:
    there is no return from this function
    """
    pickle.dump(model, open('{1}/models/{0}'.format(model_filepath, os.getcwd()), 'wb'))

def main():
    """Load the data, run the model and save model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        print(Y_test.head())
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