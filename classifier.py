# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:00:10 2020

@author: NBOLLIG
"""

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.metrics import average_precision_score

# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

import sklearn
# from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve

from labeler import read_human_labels

random_state = 100
scorer = make_scorer(f1_score)
# preprocess_params = {}

models = {'RandomForestClassifier': RandomForestClassifier(),
          'LogisticRegression': LogisticRegression(),
          # 'GradientBoostingClassifier': GradientBoostingClassifier(),
          # 'SVC' : SVC()
          }

model_params = {'RandomForestClassifier': {
    'n_estimators': [100],
    'max_depth': [None, 12, 15, 17],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [5, 7, 9],
    'min_weight_fraction_leaf': [0],
    'max_features': ['auto'],
    'n_jobs': [-1],
    'random_state': [random_state],
    'class_weight': ['balanced', None]
},
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [1.e-1, 1, 1.e1, 1e2, 1e3]
    },
    'GradientBoostingClassifier': {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_leaf': [1, 3, 5],
        'max_depth': [3, 4],
        'random_state': [random_state]
    },
    'SVC':
        {'C': [1.e-05, 1.e-03, 1.e-01],
         'kernel': ['rbf', 'linear'],
         'gamma': ['auto', 1.e-01],
         'shrinking': [True, False],
         'class_weight': ['balanced'],
         'random_state': [random_state]},
}


# loading data
def load_train():
    DATA_FILE_TRAIN = '../data/label_model_output/df_train_filtered.csv'  # NB changed since not in notebooks subdirectory
    df = pd.read_csv(DATA_FILE_TRAIN)
    comments = df.Cleaned_Text.values
    y = np.round(df.prob)

    return comments, y


def load_test():
    DATA_FILE_TEST = '../data/label_model_output/df_test.csv'  # NB changed since not in notebooks subdirectory
    df = pd.read_csv(DATA_FILE_TEST)
    comments = df.Cleaned_Text.values
    y = df.human_label.values

    return comments, y


def base_model(X):
    '''Returns all 1s'''
    return np.ones(X.shape[0])


def feature_importance(model, model_name, X_test, y_test, feature_names, output_dir, fold_counter=''):
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

    fig.savefig(os.path.join(output_dir, model_name + '_feat_imp' + fold_counter + '.png'), dpi=300)


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
        # Bootstrap and calculate F1 score
        bootstrapped_sample = resample(y, model_pred)
        pred = bootstrapped_sample[0]
        lab = bootstrapped_sample[1]
        score = getattr(sklearn.metrics, score_function)(lab, pred)
        scores.append(score)

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(scores, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(scores, p))

    # Mean
    mean = np.mean(scores)

    return mean, lower, upper


def print_CI(name, val, lower, upper):
    print("%-16s %.3f%% (%.3f%%, %.3f%%)" % (name + ":", val * 100, lower * 100, upper * 100))

def print_results(y, pred, evaluate):
    f1, f1_lower, f1_upper = get_score(y, pred, 'f1_score')
    recall, recall_lower, recall_upper = get_score(y, pred, 'recall_score')
    precision, precision_lower, precision_upper = get_score(y, pred, 'precision_score')
    acc, acc_lower, acc_upper = get_score(y, pred, 'accuracy_score')

    print_CI(evaluate + ' f1', f1, f1_lower, f1_upper)
    print_CI(evaluate + ' recall', recall, recall_lower, recall_upper)
    print_CI(evaluate + ' precision', precision, precision_lower, precision_upper)
    print_CI(evaluate + ' accuracy', acc, acc_lower, acc_upper)

def run_classifier(X_train, y_train, X_test, y_test, feature_names, output_dir, fold_counter = ''):
    model_y_pred = {}
    
    if fold_counter != '':
        print("Fold " + fold_counter + "...")
    
    for model_name, model in models.items():
        print("**" * 10, model_name, "**" * 10)

        # Fit and predict
        final_model = GridSearchCV(model, model_params[model_name], scoring=scorer, n_jobs=-1, cv=5, verbose=0)
        final_model.fit(X_train, y_train)

        pred_train = final_model.predict(X_train)
        pred_test_proba = final_model.predict_proba(X_test)[:, 1]
        pred_test = final_model.predict(X_test)
        pred_base = base_model(X_test)
        
        # Store predictions
        model_y_pred[model_name] = pred_test_proba
        
        # Summarize best model
        print('BEST ESTIMATOR')
        print(final_model.best_estimator_)
        print('BEST SCORE')
        print(final_model.best_score_)
        print('BEST_PARAMS')
        print(final_model.best_params_)

        # Performance scores
        for evaluate in ['Base', 'Train', 'Test']:
            if evaluate == 'Base':
                y = y_test
                pred = pred_base
            if evaluate == 'Train':
                y = y_train
                pred = pred_train
            if evaluate == 'Test':
                y = y_test
                pred = pred_test

            print_results(y, pred, evaluate)

        # Feature importance
        feature_importance(final_model, model_name, X_test, y_test, feature_names, output_dir, fold_counter=fold_counter)

        # PR curves
        avg_precision_test = average_precision_score(y_test, pred_test)

        disp = plot_precision_recall_curve(final_model, X_test, y_test)
        disp.ax_.set_title('{0} Precision-Recall curve: '
                           'Avg. Pre.={1:0.3f}'.format(model_name, avg_precision_test), name=model_name)
        disp.ax_.get_legend().remove()
        disp.figure_.savefig(os.path.join(output_dir, model_name + '_PR' + fold_counter + '.png'), dpi=300)

    return model_y_pred, y_test

def get_features(x):
    custom_stopwords = list(text.ENGLISH_STOP_WORDS)
    custom_stopwords.remove('not')

    tfidf = TfidfVectorizer(list(x), strip_accents='unicode', decode_error='strict',
                            stop_words=custom_stopwords, min_df=2, max_df=0.7,
                            ngram_range=(1, 2))
    x_features = tfidf.fit_transform(x)
    feature_names = np.array(tfidf.get_feature_names())  # NB
    return feature_names, x_features


def main(output_dir, experiment='all'):
    assert experiment in [1, 2, 'all'], print('experiment should be either 1, 2 or "all"')
    if experiment == 1 or experiment == 'all':
        comments_train, y_train = load_train()
        comments_test, y_test = load_test()

        X = np.concatenate((comments_train, comments_test))
        feature_names, X_tfidf = get_features(X)
        X_train = X_tfidf[:len(comments_train), :]
        X_test = X_tfidf[len(comments_train):, :]

        # Run classifier using snorkel pipeline
        print("Training classifier using Snorkel pipeline ...")
        run_classifier(X_train, y_train, X_test, y_test, feature_names, output_dir)

    if experiment == 2 or experiment == 'all':
        processed_data = '../data/data_processed/data_processed.csv'
        label_dir = '../data/human_labels/'

        df = pd.read_csv(processed_data)
        human_labels = read_human_labels(label_dir)
        df = df.merge(human_labels, indicator=True, how='left', on=['record_number'])
        df = df.query('_merge=="both"').drop('_merge', axis=1)
        X = df.Cleaned_Text.values
        y = df.human_label.values
        feature_names, X_tfidf = get_features(X)
        
        # Prepare for CV loop       
        kfold = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)
        fold_num = 0
        y_true = []
        model_predictions = {}
        for model_name in models.keys():
                model_predictions[model_name] = []
        
        # CV
        for train_idx, test_idx in kfold.split(X_tfidf, y):
            X_train = X_tfidf[train_idx, :]
            X_test = X_tfidf[test_idx, :]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Run classifier
            print("Training classifier using Human labels ...")
            model_y_pred, y_test = run_classifier(X_train, y_train, X_test, y_test, feature_names, output_dir, fold_counter = str(fold_num))
            
            # Pool predictions
            y_true.extend(list(y_test))
            for model_name in models.keys():
                model_predictions[model_name].extend(list(model_y_pred[model_name]))
            
            fold_num += 1

        # Results on pooled predictions
        THRESHOLD = 0.5
        for model_name in model_predictions.keys():
            print("*** Model: " + model_name)
            y_ones = np.ones(len(model_predictions[model_name]))
            pred_proba = model_predictions[model_name]
            pred = list((np.array(pred_proba) > THRESHOLD).astype(int))
            
            # Performance results
            print_results(y_true, y_ones, evaluate="Base")
            print_results(y_true, pred, evaluate="Test")
            
            # PR curve on pooled predictions
            avg_precision_test = average_precision_score(y_true, pred)
            precision, recall, thresh = precision_recall_curve(y_true, pred_proba)                
            plt.figure()
            plt.step(recall, precision, color='k', linestyle ='-', where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(model_name + " PR pooled predictions: Avg. Pre.=%.3f" % (avg_precision_test,))
            plt.savefig(os.path.join(output_dir, model_name + '_PR_combined.png'), dpi = 300)


if __name__ == "__main__":
    output_dir = '../data/classifier_output/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(output_dir, experiment=2)
