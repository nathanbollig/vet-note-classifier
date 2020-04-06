# -*- coding: utf-8 -*-
"""
Labeling functions that examine UWVC discharges for rule-outs.

Each function takes a single Pandas Series object (row of a dataframe) and outputs 
one of these labels:
    1 - Suspected Addison's
    0 - Not suspected Addison's
   -1 - Abstain

Created on Fri Mar 27 11:19:04 2020

@author: NBOLLIG
"""

from LFs.LF_utils import make_keyword_callable

# =============================================================================
# Keyword searches
# =============================================================================

healthy_keywords = make_keyword_callable(keywords = ['healthy'], label=0)
kidney_keywords = make_keyword_callable(keywords = ['acute kidney', 'aki', 'akd'], label=0)
parasite_keywords = make_keyword_callable(keywords = ['parasite', 'worm', 'roundworm', 'toxocara', 
                                                      'hookworm', 'ancyclostoma', 'uncinaria', 
                                                      'whipworm', 'trichuris', 'tapeworm', 'cestod', 
                                                      'taenia', 'echinococcus', 'spirometra', 
                                                      'diphyllobothrium', 'mesocestoides', 
                                                      'dipylidium', 'fluke', 'trematode', 'giardia'], label=0)
liver_keywords = make_keyword_callable(keywords = ['liver failure', 'shunt', 'hepatitis', 'cancer', 'lepto', 'hepatitis', 'copper', 'cholangio', 'diabetes'], label=0)
panc_keywords = make_keyword_callable(keywords = ['pancreatitis', 'pancreatic'], label=0)
toxin_keywords = make_keyword_callable(keywords = ['toxin', 'heavy metal', 'herbicide', 'fungicide', 
                                                   'insecticide', 'rodent', 'aflatoxin', 
                                                   'amanita', 'cycad', 'sago palm', 'algae'], label=0)
effusion_keywords = make_keyword_callable(keywords = ['effusi'], label=0)
primary_GI_keywords = make_keyword_callable(keywords = ['megaesoph', 'dilatation', 'gastritis', 'foreign body', 'intuss', 
                                                        'enteritis', 'colitis'], label=0)