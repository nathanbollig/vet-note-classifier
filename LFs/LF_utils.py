# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:19:23 2020

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