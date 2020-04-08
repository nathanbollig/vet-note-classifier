# -*- coding: utf-8 -*-
"""
Pulls processed data, applies LFs, saves filtered training set dataframe and labels.

Created on Wed Mar 18 09:30:34 2020

@author: NBOLLIG
"""

from LFs.LF_GI import *
from LFs.LF_lab_tests import *
from LFs.LF_post_hoc import *
from LFs.LF_rule_outs import *
from LFs.LF_utils import make_lfs_list

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel

import pandas as pd
import numpy as np
import os

from sklearn.metrics import f1_score

# =============================================================================
# TODO: Add new LFs to list!!!
# =============================================================================
"""
The label functions are imported as callables that return 1, 0, or -1, and sorted
into the appropriate categories below.
"""

post_hoc_callables = [
    dx_keywords,
    test_keywords,
    tx_keywords,
]

GI_callables = [
    GI_keywords_1,
    GI_keywords_2,
    GI_keywords_3,
    GI_keywords_4,
    GI_keywords_5,
    GI_keywords_6,
    GI_keywords_7,
    GI_keywords_8,
    GI_keywords_9,
]

rule_out_callables = [
    healthy_keywords,
    kidney_keywords,
    parasite_keywords,
    liver_keywords,
    panc_keywords,
    toxin_keywords,
    effusion_keywords,
    primary_GI_keywords,
]

lab_callables = [
    hyponatremia_keywords,
    hyperkalemia_keywords,
]

lfs = make_lfs_list(post_hoc_callables, 
                  GI_callables, 
                  rule_out_callables, 
                  lab_callables)

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

    return pd.concat(dataframes).drop_duplicates(subset = 'record_number')


"""
Pulls together df_test, label matrix, pred probability, and human labels 
into error analysis spreadsheet.
"""
def error_analysis(df_test, L_test, lfs, preds, lab_test, output_dir):
    df = df_test.copy()
    assert(df.shape[0] == L_test.shape[0])
    assert(len(lfs) == L_test.shape[1])
    assert(df.shape[0] == len(lab_test))
    
    # Add LF results
    for i in range(L_test.shape[1]):
        colname = lfs[i].name
        df[colname] = pd.Series(L_test[:, i], index=df.index)

    # Add prediction
    df['pred'] = pd.Series(preds, index=df.index)

    # Add human label
    df['human_label'] = pd.Series(lab_test, index=df.index)
    
    # Save
    path = os.path.join(output_dir, 'error_analysis.csv')
    df.to_csv(path, index = False)

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
        error_analysis.csv
        
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
    for metric in ['recall', 'precision', 'f1', 'accuracy']:
        label_model_acc = label_model.score(L=L_test, Y=lab_test, metrics=[metric], tie_break_policy="random")[metric]
        print("%-15s %.2f%%" % (metric+":", label_model_acc * 100))
    
    null_f1 = f1_score(lab_test.values, np.ones((df_test.shape[0],)))
    print("%-15s %.2f%%" % ("null f1:", null_f1 * 100))
    print("%-15s %.2f%%" % ("null accuracy:", np.maximum(1-np.mean(lab_test), np.mean(lab_test)) * 100))
    
    # Save error analysis
    preds = label_model.predict_proba(L_test)
    error_analysis(df_test, L_test, lfs, preds[:,1], lab_test, output_dir)
    
    # Get labels on train
    probs_train = label_model.predict_proba(L_train)

    # Filter out unlabeled data points
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=probs_train, L=L_train)
    
    # Save filtered training set
    df_train_filtered['prob'] = probs_train_filtered[:,1]
    path = os.path.join(output_dir, 'df_train_filtered.csv')
    df_train_filtered.to_csv(path, index = False)
    
    # Save label probs
    path = os.path.join(output_dir, 'probs_train_filtered')
    np.save(path, probs_train_filtered[:,1])
    
    # Save training data set and labels
    assert len(df_test) == len(lab_test)
    df_test['human_label'] = lab_test
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
    
    
    
    
    