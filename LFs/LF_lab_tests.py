# -*- coding: utf-8 -*-
"""
Labeling functions that examine UWVC discharges for laboratory test results.

Each function takes a single Pandas Series object (row of a dataframe) and outputs 
one of these labels:
    1 - Suspected Addison's
    0 - Not suspected Addison's
   -1 - Abstain

Created on Wed Mar 18 07:30:12 2020

@author: NBOLLIG
"""
#from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
import pandas as pd

def keyword_lookup(x, keywords, label):
    if pd.isnull(x.Cleaned_Text):
        return -1
    
    if any(word in x.Cleaned_Text for word in keywords):
        return label
    
    return -1


def make_keyword_lf(keywords, label=1):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )



# =============================================================================
# Keyword searches
# =============================================================================
 
# Electrolytes
hyponatremia_keywords = make_keyword_lf(keywords = ['hyponatremia', 'low sodium', 'low na', 'sodium low', 'na low'])
hyperkalemia_keywords = make_keyword_lf(keywords = ['hyperkalemia', 'high potassium', 'high k', 'potassium high', 'k high'])
