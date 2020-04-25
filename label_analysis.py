# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:26:44 2020

@author: NBOLLIG
"""
import pandas as pd
import os
import numpy as np

# Read in all labels
# (Ensure label directory contains all labels that need to be analyzed.)
label_dir = '../data/human_labels/'

dataframes = []

for entry in os.scandir(label_dir):
   if entry.path.endswith(".csv"):
       df = pd.read_csv(entry.path)
       df = df[['record_number', 'human_label']]
       dataframes.append(df)

all_labels = pd.concat(dataframes, ignore_index = True)
unique_labels = all_labels.drop_duplicates(subset = 'record_number')

# Get subsets of duplicate labels
all_labels = all_labels.loc[all_labels['record_number'].duplicated(keep=False)].reset_index()
subsets = (all_labels).groupby('record_number')['index'].apply(tuple).tolist()
print("There are %i documents with duplicate labels" %(len(subsets)))

# Report average number of duplications per duplicated document
sizes = []
for s in subsets:
    sizes.append(len(s))
print("Avg number of duplications per duplicated document: %0.3f" %(np.mean(sizes)))

# Report numbers of duplications that occurred
m = max(sizes)
for i in range(2, m + 1):
    print("Num of duplications %i times: %i" %(i, sizes.count(i)))
    
# Find sets with any positive label
num_pos_sets = 0
idx_pos_sets = []
num_neg_sets = 0
idx_neg_sets = []

pos_idx_in_data = all_labels.loc[all_labels['human_label'] == 1]['index'].values

for i in range(len(subsets)):
    s = subsets[i]
    found = False
    for j in range(len(s)):
        if s[j] in pos_idx_in_data and found==False:
            num_pos_sets += 1
            idx_pos_sets.append(i)
            found = True
    if found==False:
        num_neg_sets += 1
        idx_neg_sets.append(i)

print("There are %i documents with at least one positive label and %i docs with all neg labels in the duplicated set" %(num_pos_sets, num_neg_sets))

# Measure agreement on sets
print("Note there is full agreement on the %i sets with all neg labels" %(num_neg_sets))

print("The following are all label sets for documents with a positive label:")

for i in idx_pos_sets:
    s = subsets[i]
    labels = []
    for j in s:
        lab = all_labels.loc[all_labels['index'] == j]['human_label'].values.item()
        labels.append(lab)
    print(labels)

        
        