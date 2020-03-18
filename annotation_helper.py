# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:29:11 2020

@author: NBOLLIG
"""
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    train_path = '../data/data_processed/data_processed.csv'
    output_dir = '../data/human_labels/'
    df_output_path = os.path.join(output_dir, 'for_labeling.csv')
    
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    
    # Load pre-processed data
    df = pd.read_csv(train_path)
    
    # Randomly sample rows and select the columns we need
    df_sample = df.sample(n=1300, random_state = 123)
    df_sample = df_sample[['record_number', 'mr', 'Cleaned_Text']]
    
    # Ensure we have 1000 records from unique patients
    df_sample = df_sample.drop_duplicates(subset=['mr'])
    df_sample = df_sample.iloc[:1000, :]
    
    # Add blank column
    df_sample["human_label"] = np.nan
    
    # Save spreadsheet for labeling
    df_sample.to_csv(df_output_path, index = False)
    
    
    