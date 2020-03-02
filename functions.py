# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:44:17 2020

@author: NBOLLIG
"""

import pandas as pd
import os

"""
Main load method. Takes directory as input and returns the data as a Pandas dataframe.
The resulting columns are as follows.
* `report_date`: date of report as Python datetime object
* `mr`: medical record number
* `sex`:
  * 0 - outlier
  * 1 - female
  * 2 - male
  * 3 - neutered male
  * 4 - spayed female
  * 5 - unknown
* `species`:
  * 0 - Avian (bird)
  * 1 - Bovine (cow)
  * 2 - Canine (dog)
  * 3 - Equine (horse)
  * 4 - Feline (cat)
  * 5 - Lapine (rabbit)
* `service`:
  * 880 - Primary Care
  * 830 - Small Animal Internal Medicine
* `comment_text`: html-formatted report text (html is reportedly non-standard)
"""

def load(directory):
    file = os.path.join(directory, "data.csv")
    data = pd.read_csv(file)
    
    # Drop columns
    data = data.drop("patient", 1)
    data = data.drop("key", 1)
    
    # Remove rows of all NaN
    data = data.dropna(how='all')
    
    # Encode date as datetime object
    data['report_date'] = pd.to_datetime(data['report_date'])
    
    # Encode sex and species
    data['sex'] = data['sex'].astype('category')
    data['species'] = data['species'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    
    return data

