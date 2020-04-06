# -*- coding: utf-8 -*-
"""
Labeling functions that examine UWVC discharges for syndromic evidence of 
disease (GI, nonspecific disease, or hypovolemic shock).

Each function takes:
    x - a single Pandas Series object (row of a dataframe)
    
Each function outputs one of these labels:
    1 - Suspected Addison's
    0 - Not suspected Addison's
   -1 - Abstain

Created on Fri Apr  3 08:38:04 2020

@author: NBOLLIG
"""

from LFs.LF_utils import make_keyword_callable

# =============================================================================
# Keyword searches
# =============================================================================

GI_keywords_1 = make_keyword_callable(keywords = ['inappetance', 'anorexia', 'not eating', 'weight loss'])
GI_keywords_2 = make_keyword_callable(keywords = ['vomiting', 'diarrhea'])
GI_keywords_3 = make_keyword_callable(keywords = ['regurg'])
GI_keywords_4 = make_keyword_callable(keywords = ['melena', 'hematochezia'])
GI_keywords_5 = make_keyword_callable(keywords = ['abdominal pain'])
GI_keywords_6 = make_keyword_callable(keywords = ['letharg', 'depression', 'depressed'])
GI_keywords_7 = make_keyword_callable(keywords = ['shake', 'shakes', 'shaking', 'weak'])
GI_keywords_8 = make_keyword_callable(keywords = ['hair loss'])
GI_keywords_9 = make_keyword_callable(keywords = ['bradycardia', 'low heart rate', 
                                            'tachycardia', 'high heart rate', 
                                            'collapse', 'hypothermia', 'low body temp', 
                                            'weak pulse', 'poor capillary refill', 
                                            'shock', 'hypovolem'])
