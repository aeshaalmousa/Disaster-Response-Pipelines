import sys
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine 
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    ''' 
    load_data function that load data from sqlite database into dataframe
    Argument: database_filepath -> file path of stored database 
    Return: 
    X -> message dataframe
    Y -> target dataframe the 36 categories
    category_names -> the columns name of each 36 categories
    '''
    #load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CleanedDf', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y , category_names


def tokenize(text):
    '''
    tokenize function that normlize, lemmatize and tokenize the message data
    Argument: text -> message data
    Return: tokenized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:    
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' 
    build_model function that build machine learning pipeline that take message column as input and tokenized and output classification results on the other 36 categories in the dataset. then make grid search for find better parameters.
    Argument: N/A
    Return: grid search model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters ={
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [3, 4]
    }
    cv = GridSearchCV(pipeline,
                      param_grid=parameters,cv = 2, n_jobs = -1, verbose = 5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    evaluate_model function that test the predicted model.
    Arguments: 
    model -> machine learning piplilne
    X_test -> message data
    Y_test -> 36 categories
    category_names -> the name of each 36 categories
    '''
    #test the model
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = category_names)
    for col in category_names: 
        print(col,':')
        print('Accuracy ',accuracy_score(Y_test[col], y_pred_pd[col])
              ,'Precision ',precision_score(Y_test[col],
                                            y_pred_pd[col],average='weighted')
              ,'Recall ',recall_score(Y_test[col],
                                      y_pred_pd[col],average='weighted') )
    pass 

def save_model(model, model_filepath):
    '''
    save_model function that save the model as pickle file
    model -> evaluated model
    model_filepath -> pickle file to save model on it
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    
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