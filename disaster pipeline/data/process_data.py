import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

print(os.getcwd())

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    categories = pd.Series(df['categories'])
    categories = categories.str.split(";", n=36, expand=True)
    row = categories.iloc[1]
    category_colnames = [i for i in map(lambda r: r[:-2], row)]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = pd.Series(categories[column]).str.replace('{0}{1}'.format(column,'-'), '')
        categories[column] = pd.to_numeric(pd.Series(categories[column]),errors='ignore')
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates('id')
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{0}/DisasterResponse.db'.format(os.getcwd()))
    df.to_sql('message', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python models\process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()