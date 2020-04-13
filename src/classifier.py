__authors__ = ['Florian BETTINI', 'Maxime LUTEL', 'Gabriel MULLER', 'Benjamin POMMIER']

#Importing modules
import numpy as np
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class Classifier:
    """The Classifier"""
    def __init__(self):
        self.model = None
        self.vect_S = None
        self.vect_C = None
        self.vect_A = None
        self.scaler = None
        self.ws = None
        self.thresh_bool = None
    
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        
        # Hyperparameters
        self.ws = 40                        # number of characters near the target term used to compute the context
        self.thresh_bool = False            # whether or not to apply different thresholds for LogisticRegression
        self.thresh_neutral = 0.45          # if the "negative" proba is higher than thresh_negative,
                                            # then y_pred becomes negative
        self.thresh_negative = 0.45         # if the proba for neutral is higher than thresh_neutral,
                                            # then y_test becomes neutral
        self.thresh_positive = 0.4          # if the proba for positive is lower than thresh_positive,
                                            # then y_test becomes negative
        C = 0.6                             # penality for LogisticRegression  
        max_features_sentence = 1000        # max features used for vect_S (Tfidf vectorizer for all sentence)
        ngram_range_sentence = (1,3)        # ngram_range (min, max) for vect_S
        max_features_context = 1000         # max features used for vect_C (Tfidf vectorizer for context)
        ngram_range_context = (1,3)         # ngram_range (min, max) for vect_C
        max_features_target = 100           # max features used for vect_A (Tfidf vectorizer for target term)
        ngram_range_target = (1,1)          # ngram_range (min, max) for vect_A
        gs_boolean = False                  # whether or not to do a grid search
        gs_parameters = {'C': [0.5, 0.6, 0.8, 1]}   # parameters grid for grid search
        
        
        # Loading train set
        df_train = pd.read_csv(trainfile, sep = '\t', header = None)
        df_train.columns = ['polarity', 'aspect_category', 'target_term', 'start:end', 'sentence']
        
        # Extract features
        X_train, self.vect_S, self.vect_C, self.vect_A = extract_features_train(df_train, self.ws,
                                                     ngram_range_sentence, max_features_sentence,
                                                     ngram_range_context, max_features_context,
                                                     ngram_range_target, max_features_target)
        
        
        # Extract labels
        y_train = df_train['polarity'].map({'positive':2, 'neutral':1, 'negative':0}).values
        
        # Model training
        self.model, self.scaler = train(X_train, y_train, C = C,
                                        grid_search = gs_boolean, parameters = gs_parameters)
        
        
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        # Loading test set
        df_test = pd.read_csv(datafile, sep = '\t', header = None)
        df_test.columns = ['polarity', 'aspect_category', 'target_term', 'start:end', 'sentence']
        
        # Extract features
        X_test = extract_features_test(df_test, self.ws, self.vect_S, self.vect_C, self.vect_A)
        
        # Scaling features
        X_test = pd.DataFrame(self.scaler.transform(X_test.astype('float')), columns = X_test.columns)
        
        # Predict labels
        y_pred = self.model.predict(X_test)
        
        # modify thresholds
        if self.thresh_bool:
            probas = self.model.predict_proba(X_test)            
            y_pred_thresh = np.where(probas[:,0] > self.thresh_negative, 0, y_pred)
            y_pred_thresh = np.where(probas[:,1] > self.thresh_neutral, 1, y_pred_thresh)
            y_pred_thresh = np.where(probas[:,2] < self.thresh_positive, 0, y_pred_thresh)
            y_pred = y_pred_thresh
            
        slabels = mapping(y_pred)
        return slabels
    

## Main Functions

def extract_features_train(df_train, ws, ng_S, max_features_S, ng_C, max_features_C, ng_A, max_features_A):
    """Extract all features needed to train the model
    Returns a dataframe X with all features, and the 3 vectorizers used
    """
    # Extract context (eg, the part of the sentence near the target term)
    df_train['context'] = df_train[['start:end', 'sentence']].apply(lambda x: extract_context(x, ws), axis = 1)
    
    # feature extraction n°1
    df_train['length'] = df_train.sentence.str.len() # adding sentence length
    df_train['nb_upper'] = df_train.sentence.apply(count_upper_chars) # number of upper letter
    df_train['proportion_upper'] = df_train.nb_upper / df_train.length # proportion of upper letter
    
    # cleaning
    df_train.drop(['start:end', 'aspect_category'], axis = 1, inplace = True)
    
    # feature extraction n°2: TfIdf vectorizer
    # All sentence (S)
    X_sentence, vect_S = build_tfidf_vectorizer(df_train.sentence, ng_S, max_features_S)
    X_sentence = change_columns(X_sentence, 'S - ') # S for Sentence
    
    # Context (C)
    X_context, vect_C = build_tfidf_vectorizer(df_train.context, ng_C, max_features_C)
    X_context = change_columns(X_context, 'C - ') # C for Context
    
    # Target term (A)
    X_target, vect_A = build_tfidf_vectorizer(df_train.target_term, ng_A, max_features_A)
    X_target = change_columns(X_target, 'A - ') # A for Aspect
    
    # Merging all features
    X = pd.concat([df_train[['length', 'nb_upper', 'proportion_upper']], X_sentence], axis = 1)
    X = pd.concat([X, X_context], axis = 1)
    X = pd.concat([X, X_target], axis = 1)
    
    return X, vect_S, vect_C, vect_A



def train(X_train, y_train, C=1, grid_search = False, parameters = {'C': [0.5, 1, 1.5]}):
    """Train a LogisticRegression model
    Returns both model and scaler
    """
    # scaler
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float')), columns = X_train.columns)
    
    # model
    log_reg = LogisticRegression(solver = 'lbfgs', penalty = 'l2',
                                 C = C, multi_class = 'auto', max_iter=1000)

    # gridsearch
    if grid_search:
        param_grid = parameters
        gs = GridSearchCV(estimator = log_reg, param_grid = param_grid, cv = 5, verbose = 1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
    else:
        log_reg.fit(X_train, y_train)
        model = log_reg
    
    return model, scaler



def extract_features_test(df_test, ws, vect_S, vect_C, vect_A):
    """Extract all features needed to predict labels
    Returns a dataframe X with all features
    """
    # Extract context (eg, the part of the sentence near the target term)
    df_test['context'] = df_test[['start:end', 'sentence']].apply(lambda x: extract_context(x, ws), axis = 1)
    
    # feature extraction n°1
    df_test['length'] = df_test.sentence.str.len() # adding sentence length
    df_test['nb_upper'] = df_test.sentence.apply(count_upper_chars) # number of upper letter
    df_test['proportion_upper'] = df_test.nb_upper / df_test.length # proportion of upper letter
    
    # cleaning
    df_test.drop(['start:end', 'aspect_category'], axis = 1, inplace = True)
    
    # feature extraction n°2: TfIdf vectorizer
    # All sentence (S)
    X_sentence = use_tfidf_vectorizer(df_test.sentence, vect_S)
    X_sentence = change_columns(X_sentence, 'S - ') # S for Sentence
    
    # Context (C)
    X_context = use_tfidf_vectorizer(df_test.context, vect_C)
    X_context = change_columns(X_context, 'C - ') # C for Context
    
    # Target term (A)
    X_target = use_tfidf_vectorizer(df_test.target_term, vect_A)
    X_target = change_columns(X_target, 'A - ') # A for Aspect
    
    # Merging all features
    X = pd.concat([df_test[['length', 'nb_upper', 'proportion_upper']], X_sentence], axis = 1)
    X = pd.concat([X, X_context], axis = 1)
    X = pd.concat([X, X_target], axis = 1)
    
    return X



## Utils
        
def extract_context(row, ws):
    """Extract all words near a target term within a sentence
    Returns all words near the target term
    """
    start = max(0, int(row['start:end'].split(':')[0]) - ws)
    end = min(len(row['sentence']), int(row['start:end'].split(':')[1]) + ws)
    sentence = row['sentence'][start:end]
    if (start == 0) & (end == len(row['sentence'])):
        return row['sentence'][start:end]
    elif (end == len(row['sentence'])):
        if row['sentence'][start-1] != ' ':
            new_start = sentence.find(' ')
            return sentence[new_start+1:]
        else:
            return sentence
    elif (start == 0):
        if row['sentence'][end] != ' ':
            new_end = sentence[::-1].find(' ')
            return sentence[:len(sentence)-new_end-1]
        else:
            return sentence
    else:
        if (row['sentence'][start-1] != ' ') & (row['sentence'][end] != ' '):
            new_start = sentence.find(' ')
            new_end = sentence[::-1].find(' ')
            return sentence[new_start+1:len(sentence)-new_end-1]
        elif row['sentence'][end] != ' ':
            new_end = sentence[::-1].find(' ')
            return sentence[:len(sentence)-new_end-1]
        elif row['sentence'][start-1] != ' ':
            new_start = sentence.find(' ')
            return sentence[new_start+1:]
        else:
            return sentence
        
        
def count_upper_chars(sentence):
    """Count the number for characters within a string that are in upper case"""
    return sum(map(str.isupper, sentence))


def build_tfidf_vectorizer(series, ngram_range, max_features):
    """Build a Tfidf vectorizer when training a model
    Returns the encoded dataframe and the tfidf vectorizer
    """
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=ngram_range, max_features=max_features,
                           token_pattern=r'\b[^\d\W][^\d\W]+\b')
    vect.fit(series)
    X = vect.transform(series)
    X = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    return X, vect


def use_tfidf_vectorizer(series, vect):
    """Use a pre-fitted Tfidf vectorizer when testing the model
    Returns the encoded dataframe
    """
    X = vect.transform(series)
    X = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    return X


def change_columns(X, tag):
    """Change the name of the columns in X by adding a tag in front of each word
    Returns the new list of columns
    """
    columns = []
    for name in X.columns:
        columns.append(tag + name)
    X.columns = columns
    return X


def mapping(y):
    """Format all labels (0, 1, 2) to (negative, neutral, positive)
    Returns the modified labels
    """
    labels = np.where(y == 0, 'negative', y)
    labels = np.where(y == 1, 'neutral', labels)
    labels = np.where(y == 2, 'positive', labels)
    return labels