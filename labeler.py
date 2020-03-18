# -*- coding: utf-8 -*-
"""
Pulls processed data, applies LFs, saves filtered training set dataframe and labels.

Created on Wed Mar 18 09:30:34 2020

@author: NBOLLIG
"""

from LFs.LF_lab_tests import *
from LFs.LF_post_hoc import *

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
]

# =============================================================================

if __name__ == '__main__':
    train_path = '../data/data_processed/data_processed.csv'
    output_dir = '../data/label_model_output/'
    
    df = pd.read_csv(train_path)
    
    # TODO: as below
    # Create df_train by removing any patients for which there is a human label. 
    # Create df_test of all human-labeled notes. 
    # Should be dynamically responsive to whatever annotated data we have up to this point.
    
    
    # Generate label matrix
    # TODO: apply to df_train and df_test to get L_train and L_test
    L_train = PandasLFApplier(lfs=lfs).apply(df=df)
    
    # Summarize LFs
    output = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(output)
    
    # Create label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    
    # TODO: Evaluate the label model using labeled test set
    
    # Get labels on train
    probs_train = label_model.predict_proba(L_train)

    # Filter out unlabeled data points
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df, y=probs_train, L=L_train)
    
    # Save filtered training set
    df_output_path = os.path.join(output_dir, 'df_train_filtered.csv')
    df_train_filtered.to_csv(df_output_path, index = False)
    
    # Save label probs
    probs_output_path = os.path.join(output_dir, 'probs_train_filtered')
    np.save(probs_output_path, probs_train_filtered)
    
    
    
    