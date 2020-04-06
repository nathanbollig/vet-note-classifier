# -*- coding: utf-8 -*-
"""
Labeling functions that examine UWVC discharges for post-hoc evidence of a clinical suspicion.

Each function takes a single Pandas Series object (row of a dataframe) and outputs 
one of these labels:
    1 - Suspected Addison's
    0 - Not suspected Addison's
   -1 - Abstain

Created on Wed Mar 18 09:28:26 2020

@author: NBOLLIG
"""

from LFs.LF_utils import make_keyword_callable

# =============================================================================
# Keyword searches
# =============================================================================

# Direct mentions of diagnosis
dx_keywords = make_keyword_callable(keywords = ['hypoadrenocorticism', 'addisons', 'addison\'s'], section = 'Diagnoses')

#def no_dx(x):
#    if (dx_keywords(x) == -1):
#        return 0
#    else:
#        return -1

# Laborary test was mentioned
test_keywords = make_keyword_callable(keywords = ['baseline cort', 'baseline cortisol', 'acth'], section = 'Diagnostic_Test')

# Treatment was mentioned
tx_keywords = make_keyword_callable(keywords = ['desoxycorticosterone pivalate', 'desoxycorticosterone', 'fludrocortisone acetate', 'fludrocortisone', 'florinef', 'docp'], section = 'Treatment')
    