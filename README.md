# vet-note-classifier
Applying weak supervision to train a classifier of veterinary clinical notes

## Raw Data Columns
* `report_date`: date string as mm/dd/YYYY
* `patient`: name (last, first)
* `mr`: medical record number
* `sex`:
  * F - female
  * M - male
  * N - neutered male
  * S - spayed female
  * U - unknown
* `species`:
  * Avian (bird)
  * Bovine (cow)
  * Canine (dog)
  * Equine (horse)
  * Feline (cat)
  * Lapine (rabbit)
* `service`:
  * 880 - Primary Care
  * 830 - Small Animal Internal Medicine
* `comment_text`: html-formatted report text (html is reportedly non-standard)

## Pipeline (as of 3/18)
Note: The function load() in functions.py was previously used and may be useful at some point for encoding string features prior to model training.

### Data Processing
STEP 1: With respect to working directory, raw data is located at '../data/data_raw/data.csv'.

STEP 2: Run code from the second half of the explore.ipynb Jupyter notebook, which creates '../data/data_processed/data_processed.csv'. The processed data output is the input for labeling.

### Labeling
STEP 3: `labeler.py` script loads processed data, applies the labeling functions, creates a label model, and outputs labels to '../data/labels/label_model_output.csv'.

## Labeling Functions
Each labeling function (LF) takes a Pandas Series object (row of a dataframe) and outputs either 1 (suspected Addison's), 0 (not suspected Addison's), or -1 (abstain).

LFs are organized into several files in the LFs subdirectory.
* `LF_lab_tests.py` contains labeling functions that label based on laboratory test information
* `LF_post_hoc.py` contains labeling functions evaluating for the presence of post-hoc evidence of clinical suspicion (diagnostic tests performed, treatments given, etc.)

## Instructions for Manual Annotation
