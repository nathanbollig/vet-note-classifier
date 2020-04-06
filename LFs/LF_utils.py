# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:19:23 2020

@author: NBOLLIG
"""

#from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
import pandas as pd
import nltk

def keyword_lookup(x, keywords, label, look_for_negative_polarity=False, 
                   section = 'Cleaned_Text', label_when_keywords_absent=False):
    text = x[section]
#    negative_words = ['not', 'no', 'absent', 'none']
    
    if pd.isnull(text):
        return -1
    if label_when_keywords_absent == False:
        if look_for_negative_polarity == False:
            if any(word in text for word in keywords):
                return label
#        else:
#            word_list = nltk.word_tokenize(text)
#            for i in range(len(word_list)): # Very slow - any possible optimization?
#                for keyword in keywords:
#                    if keyword in word_list[i]:
#                        if any(negation in word_list[i-15:i+5] for negation in negative_words):
#                            return -1
#                        else:
#                            return label
    else:
        if not any(word in text for word in keywords):
            return label
                    
    return -1

def make_keyword_callable(keywords, label=1, look_for_negative_polarity=False, 
                          section='Cleaned_Text', label_when_keywords_absent=False):
    def function(x, keywords=keywords, label=label, look_for_negative_polarity=look_for_negative_polarity, 
                 section='Cleaned_Text', label_when_keywords_absent=False):
        return keyword_lookup(x, keywords=keywords, label=label, look_for_negative_polarity=look_for_negative_polarity, 
                              section='Cleaned_Text', label_when_keywords_absent=False)
    function.__name__ = keywords[0]
    return function



"""
This function takes callables sorted into categories, forms a complete list of 
label functions, then wraps all label functions in a snorkel LabelFunction object.

"""
def make_lfs_list(post_hoc_callables, 
                  GI_callables, 
                  rule_out_callables, 
                  lab_callables):
    lfs = []
    
    for f in post_hoc_callables:
        lfs.append(f)
    
    for f in rule_out_callables:
        lfs.append(f)
        
    for f in GI_callables:
        lfs.append(f)
    
    for f in lab_callables:
        lfs.append(f)
        
#    """
#    Returns 1 if any rule out condition is met, 0 otherwise.
#    """
#    def rule_out(x):
#        for f in rule_out_callables:
#            if f(x)==0:
#                return 1
#        return 0
#    
#    """
#    1 if any GI callable labels 1, -1 otherwise
#    """
#    def any_GI(x):
#        for f in GI_callables:
#            if f(x)==1:
#                return 1
#        return -1
#    
#    """
#    1 if any lab callable labels 1, -1 otherwise
#    """
#    def any_lab(x):
#        for f in lab_callables:
#            if f(x)==1:
#                return 1
#        return -1
#    
#    """
#    Form LFs
#    """
#    for f in GI_callables:
#        def new_lf(x, f=f):
#            if f(x) == 1 and rule_out(x) == 0:
#                return 1
#            else: 
#                return -1
#        new_lf.__name__ = f.__name__
#        lfs.append(new_lf)
#        
#        def GI_any_lab(x, new_lf=new_lf):
#            if new_lf(x) == 0:
#                return 0
#            elif new_lf(x) == 1 and any_lab(x) == 1:
#                return 1
#            else: 
#                return -1
#        GI_any_lab.__name__ = f.__name__+'_lab'
#        lfs.append(GI_any_lab)
#        
#    
#    for f in lab_callables:
#        def new_lf(x, f=f):
#            if f(x) == 1 and rule_out(x) == 0:
#                return 1
#            else: 
#                return -1
#        new_lf.__name__ = f.__name__
#        lfs.append(new_lf)
#        
#        def lab_any_GI(x, new_lf=new_lf):
#            if new_lf(x) == 0:
#                return 0
#            elif new_lf(x) == 1 and any_GI(x) == 1:
#                return 1
#            else: 
#                return -1
#        lab_any_GI.__name__ = f.__name__+'_GI'
#        lfs.append(lab_any_GI)
    
    wrapped_lfs = []
    for lf in lfs:
        wrapped_lfs.append(LabelingFunction(name=lf.__name__, f=lf))
    
    return wrapped_lfs
    
    
    
    
    
    
    
    
    
    
    
    
    