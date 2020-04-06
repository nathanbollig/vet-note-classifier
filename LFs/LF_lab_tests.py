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

from LFs.LF_utils import make_keyword_callable

# =============================================================================
# Keyword searches
# =============================================================================
 
# Electrolytes
hyponatremia_keywords = make_keyword_callable(keywords = ['hyponatremia', 'low sodium', 'low na', 'sodium low', 'na low'], section = 'Diagnostic_Test')
hyperkalemia_keywords = make_keyword_callable(keywords = ['hyperkalemia', 'high potassium', 'high k', 'potassium high', 'k high'], section = 'Diagnostic_Test')
