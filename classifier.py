# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:00:10 2020

@author: NBOLLIG
"""

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV
#from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score

#from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

import sklearn
#from sklearn.inspection import permutation_importance
from sklearn.utils import resample

random_state = 100
scorer = make_scorer(f1_score)
#preprocess_params = {}

models = {'RandomForestClassifier': RandomForestClassifier(),
         'LogisticRegression': LogisticRegression(),
         #'SVC' : SVC()
         }

model_params = {'RandomForestClassifier':
              {'n_estimators':[10],
               'max_depth': [None, 15, 20, 25, 30],
              'min_samples_split' : [2, 3],
              'min_samples_leaf': [1, 3, 5, 7],
              'min_weight_fraction_leaf': [0],
              'max_features': ['auto'],
              'n_jobs': [-1],
              'random_state':[random_state],
              'class_weight':['balanced', None]
              },
            'LogisticRegression':{
                'penalty':['l1','l2'],
                'C':[1.e-1, 1, 1.e1]
            },
            'SVC':
              {'C':[1.e-05, 1.e-03, 1.e-01],
              'kernel':['rbf', 'linear'],
              'gamma' : ['auto', 1.e-01],
              'shrinking':[True, False],
              'class_weight':['balanced'],
              'random_state':[random_state]},
         }

#loading data
def load_train():
    DATA_FILE_TRAIN = '../data/label_model_output/df_train_filtered.csv' #NB changed since not in notebooks subdirectory
    df = pd.read_csv(DATA_FILE_TRAIN)
    comments = df.Cleaned_Text.values
    y = np.round(df.prob)
    
    return comments, y

def load_test():
    DATA_FILE_TEST = '../data/label_model_output/df_test.csv' #NB changed since not in notebooks subdirectory
    df = pd.read_csv(DATA_FILE_TEST)
    comments = df.Cleaned_Text.values
    y = df.human_label.values
    
    return comments, y

def base_model(X):
    '''Returns all 1s'''
    return np.ones(X.shape[0])

def feature_importance(model, model_name, X_test, y_test, feature_names, output_dir):
    
    if model_name == 'RandomForestClassifier':
#        result = permutation_importance(model, X_test.toarray(), y_test, n_repeats=10, random_state=42, n_jobs=2)
#        sorted_idx = result.importances_mean.argsort()
#        
#        fig, ax = plt.subplots()
#        ax.boxplot(result.importances[sorted_idx].T,
#                   vert=False, labels=X_test.columns[sorted_idx])
#        ax.set_title("Permutation Importances (test set)")
#        fig.tight_layout()
#        plt.show()
        
        tree_feature_importances = model.best_estimator_.feature_importances_
        indices = tree_feature_importances.argsort()[::-1]
        fig, ax = plt.subplots()
        names = feature_names[indices[:15]]
        y_ticks = np.arange(0, len(names))
        ax.barh(y_ticks, tree_feature_importances[indices[:15]])
        ax.set_yticklabels(names)
        ax.set_yticks(y_ticks)
        ax.set_title("Random Forest Feature Importances (MDI)")
        fig.tight_layout()
        plt.show()
    
    if model_name == 'LogisticRegression':
        coefs = model.best_estimator_.coef_[0]
        indices = np.argsort(coefs)[::-1]
        fig, ax = plt.subplots()
        
        names = feature_names[indices[:15]]
        y_ticks = np.arange(0, len(names))
        ax.barh(y_ticks, coefs[indices[:15]], color="r", align="center")
        ax.set_yticklabels(names)
        ax.set_yticks(y_ticks)
        
        ax.set_title("Feature importances (Logistic Regression)")
        fig.tight_layout()
        plt.show()
    
    fig.savefig(os.path.join(output_dir, model_name+'_feat_imp.png'), dpi = 300)

"""
Get performance score and confidence interval using bootstrapping.

Inputs:
    pred - predicted labels from model
    y - true labels
    score_function - scikit-learn score function

Returns:
    metric, lower limit of 95% CI, upper limit of 95% CI
"""
def get_score(model_pred, y, score_function):    
    scores = []

    for i in range(2000):
        #Bootstrap and calculate F1 score
        bootstrapped_sample = resample(y, model_pred)
        pred = bootstrapped_sample[0]
        lab = bootstrapped_sample[1]
        score = getattr(sklearn.metrics, score_function)(lab, pred)
        scores.append(score)
      
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(scores, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(scores, p))
    
    # Mean
    mean = np.mean(scores)
    
    return mean, lower, upper

def print_CI(name, val, lower, upper):    
    print("%-16s %.3f%% (%.3f%%, %.3f%%)" % (name+":", val * 100, lower * 100, upper * 100))

def run_classifier(X_train, y_train, X_test, y_test, feature_names, output_dir):
    for model_name, model in models.items():
        print("**"*10, model_name, "**"*10)
        
        # Fit and predict
        final_model = GridSearchCV(model, model_params[model_name],scoring=scorer, n_jobs= -1, cv = 5, verbose= 0)
        final_model.fit(X_train, y_train)

        pred_train = final_model.predict(X_train)
        pred_test = final_model.predict(X_test)
        pred_base = base_model(X_test)

        # Summarize best model
        print('BEST ESTIMATOR')
        print(final_model.best_estimator_)
        print('BEST SCORE')
        print(final_model.best_score_)
        print('BEST_PARAMS')
        print(final_model.best_params_)
        
        # Performance scores
        for evaluate in ['Base', 'Train', 'Test']:
            if evaluate=='Base':
                y = y_test
                pred = pred_base
            if evaluate=='Train':
                y = y_train
                pred = pred_train
            if evaluate=='Test':
                y = y_test
                pred = pred_test
            
            f1, f1_lower, f1_upper = get_score(y, pred, 'f1_score')
            recall, recall_lower, recall_upper = get_score(y, pred, 'recall_score')
            precision, precision_lower, precision_upper = get_score(y, pred, 'precision_score')
            acc, acc_lower, acc_upper = get_score(y, pred, 'accuracy_score')
            
            print_CI(evaluate+' f1', f1, f1_lower, f1_upper)
            print_CI(evaluate+' recall', recall, recall_lower, recall_upper)
            print_CI(evaluate+' precision', precision, precision_lower, precision_upper)
            print_CI(evaluate+' accuracy', acc, acc_lower, acc_upper)
        
        # Feature importance
        feature_importance(final_model, model_name, X_test, y_test, feature_names, output_dir)
        
        # PR curves
        avg_precision_test = average_precision_score(y_test, pred_test)
        
        disp = plot_precision_recall_curve(final_model, X_test, y_test)
        disp.ax_.set_title('Precision-Recall curve: '
                   'AP={0:0.2f}'.format(avg_precision_test), name = model_name)
        disp.ax_.get_legend().remove() #NB: remove legend and add space to plot title
        disp.figure_.savefig(os.path.join(output_dir, model_name+'_PR.png'), dpi = 300)

def main(output_dir):
    comments_train, y_train = load_train()
    comments_test, y_test = load_test()
    
    tfidf = TfidfVectorizer(list(comments_train), strip_accents='unicode', decode_error= 'strict',
                        stop_words = 'english',
                       ngram_range=(1,1))

    X_train = tfidf.fit_transform(comments_train)
    X_test = tfidf.transform(comments_test)
    feature_names = np.array(tfidf.get_feature_names()) # NB
    
    # Run classifier using snorkel pipeline
    print("Training classifier using Snorkel pipeline ...")
    run_classifier(X_train, y_train, X_test, y_test, feature_names, output_dir)
    
    # TODO: Run CV experiment using human labels for training
        
if __name__ == "__main__":
    output_dir = '../data/classifier_output/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    main(output_dir)