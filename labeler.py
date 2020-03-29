# -*- coding: utf-8 -*-
"""
Pulls processed data, applies LFs, saves filtered training set dataframe and labels.

Created on Wed Mar 18 09:30:34 2020

@author: NBOLLIG
"""

from LFs.LF_lab_tests import *
from LFs.LF_post_hoc import *
from LFs.LF_rule_outs import *

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel

import pandas as pd
import numpy as np
import os

# =============================================================================
# TODO: Add new LFs to list!!!
# =============================================================================

lfs = [
    hyponatremia_keywords,
    hyperkalemia_keywords,
    dx_keywords,
    test_keywords,
    tx_keywords,
    healthy_keywords,
]

# =============================================================================

"""
Reads human labels from all .csv files in label_dir, where the spreadsheets
are required to have 'record_number' and 'human_label' columns. Combines labels
into a single Pandas dataframe
"""
def read_human_labels(label_dir):
    dataframes = []
    
    for entry in os.scandir(label_dir):
       if entry.path.endswith(".csv"):
           df = pd.read_csv(entry.path)
           df = df[['record_number', 'human_label']]
           dataframes.append(df)

    return pd.concat(dataframes)   

"""
loads processed data, applies the labeling functions, creates a label model

Output filtered training data (where at least one LF did not abstain) as Pandas
dataframe and corresponding label model output probabilities as Numpy array. 
Similarly output test set data and labels. Analysis of LF on train and test set
is also output.

Parameters:
    train_path - path for processed data for training
    
    output_dir - directory to save output (files as below)
        df_train_filtered.csv
        probs_train_filtered.npy
        df_test.csv
        lab_test.npy
        LF_analysis_train.csv
        LF_analysis_test.csv
        
    label_dir - directory containing .csv files of human labels    

"""
def main(train_path, output_dir, label_dir):
    # Get all data
    df = pd.read_csv(train_path)
    
    # Get human labels
    human_labels = read_human_labels(label_dir)
    
    # df_test and lab_test: the set of all human-labeled notes, and their labels
    df_test = df.merge(human_labels, on=['record_number'])
    lab_test = df_test.human_label
    del df_test['human_label']
    
    # df_train: formed by removing all patients from df with a human-labeled note
    df_train = df.merge(df_test.mr, indicator=True, how='left', on = ['mr'])
    df_train = df_train.query('_merge=="left_only"').drop('_merge', axis=1)   
    
    # Generate label matrix
    L_train = PandasLFApplier(lfs=lfs).apply(df=df_train)
    L_test = PandasLFApplier(lfs=lfs).apply(df=df_test)
    
    # Summarize LFs
    output_train = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    #print(output_train)
    output_test  = LFAnalysis(L=L_test, lfs=lfs).lf_summary(Y = lab_test.values)
    #print(output_test)
    
    # Save LF analysis
    path = os.path.join(output_dir, 'LF_analysis_train.csv')
    output_train.to_csv(path, index = True)
    path = os.path.join(output_dir, 'LF_analysis_test.csv')
    output_test.to_csv(path, index = True)
    
    # Create label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    
    # Evaluate the label model using labeled test set
    metric = 'f1'
    label_model_acc = label_model.score(L=L_test, Y=lab_test, metrics=[metric], tie_break_policy="random")[metric]
    
    print("Label model %s score is %.2f%% ..." % (metric, label_model_acc * 100))
    
    # Get labels on train
    probs_train = label_model.predict_proba(L_train)

    # Filter out unlabeled data points
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=probs_train, L=L_train)
    
    # Save filtered training set
    path = os.path.join(output_dir, 'df_train_filtered.csv')
    df_train_filtered.to_csv(path, index = False)
    
    # Save label probs
    path = os.path.join(output_dir, 'probs_train_filtered')
    np.save(path, probs_train_filtered[:,1])
    
    # Save training data set and labels
    path = os.path.join(output_dir, 'df_test.csv')
    df_test.to_csv(path, index = False)
    path = os.path.join(output_dir, 'lab_test')
    np.save(path, lab_test)

if __name__ == '__main__':
    train_path = '../data/data_processed/data_processed.csv'
    output_dir = '../data/label_model_output/'
    label_dir = '../data/human_labels/'
    
    main(train_path, output_dir, label_dir)
    
    print("Finished saving label model analysis and output to '%s' ..." % (output_dir))
    
    
    
    
    