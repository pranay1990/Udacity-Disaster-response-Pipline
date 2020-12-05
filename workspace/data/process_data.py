import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message and categories csv file.
    args:
        messages_filepath: it is the path to the messages.csv file
        categories_filepath: it is the path to the categories.csv file
    Returns:
        df: the inner joined dataset of the categories and messages datsets 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how='inner',on=['id'])
    return df


def clean_data(df):
    """
    Here we clean the data
    args:
        df: uncleaned data
    Returns:
        df: cleaned data
    """
    categories = df['categories'].str.split(pat=';',expand=True) 
    row = categories.loc[0,:]
    
    category_colnames = []
    for i in row:
        category_colnames.append(i[0:i.find('-')])
        
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x)-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # converting number greater than 1 to 1
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
        
    df=df.drop(['categories'], axis=1)
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(subset=['message'], keep='first')
    return df


def save_data(df, database_filename):
    """
    Here we save the cleaned data in the SQL database form.
    args:
        df: cleaned data
    Returns:
        database_filename: the place where the SQL database to be stored
    """
    engine=f'sqlite:///{database_filename}'
    df.to_sql('Disaster_Response_Database', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()