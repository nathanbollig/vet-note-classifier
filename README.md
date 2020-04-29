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

## Weak Supervision Pipeline

### Data Processing
STEP 1: With respect to working directory, raw data is located at '../data/data_raw/data.csv'.

STEP 2: Run `data_cleaning.py`, which creates '../data/data_processed/data_processed.csv'. The processed data output is the input for labeling.

### Label Model
STEP 3: `labeler.py` script loads processed data, applies the labeling functions, creates a label model.

STEP 4: Output the following data to '../data/label_model_output/'.
* `df_train_filtered.csv` - filtered training data (where at least one LF did not abstain)
* `probs_train_filtered.npy` - corresponding label model output probabilities (probability of a 1)
* `df_test.csv` - test data, based on human-provided labels in '../data/human_labels/' (human_label column also included)
* `lab_test.npy` - binary test labels provided by a human (redundant)
* `LF_analysis_train.csv` - analysis of labeling functions on the training data
* `LF_analysis_test.csv` - analysis of labeling functions on the test data

### Train Classifier
STEP 5: Run `classifier.py`. It pulls in output from label model and manual labels, then trains and evaluates a classifier on the test set. Saves results to '../data/classifier_output/'. 

## Main Experiments
Ensure that the most up-to-date `data_cleaning.py` was run on your machine before running either experiment.

### Experiment 1: Weak Supervision Pipeline
* For evaluating on tuning set: ensure that '../data/human_labels/' contains LS_199.csv and LS_266.csv. Run `labeler.py` and `classifier.py` as described in weak supervision pipeline.
* For evaluating on the test set: ensure that '../data/human_labels/' contains LS_534.csv. Run `labeler.py` and `classifier.py` as described in weak supervision pipeline.

### Experiment 2: Training on Human Labels Only
1. Ensure that '../data/human_labels/' contains LS_199.csv, LS_266.csv, and LS_534.csv.
2. Modify `classifier.py` to call main with parameter 'experiment = 2'.

## Labeling Functions
Each labeling function (LF) takes a Pandas Series object (row of a dataframe) and outputs either 1 (suspected Addison's), 0 (not suspected Addison's), or -1 (abstain).

LFs are organized into several files in the LFs subdirectory.
* `LF_GI.py` contains labeling functions that look for clinical signs
* `LF_lab_tests.py` contains labeling functions that label based on laboratory test information
* `LF_post_hoc.py` contains labeling functions evaluating for the presence of post-hoc evidence of clinical suspicion (diagnostic tests performed, treatments given, etc.)
* `LF_rule_outs.py` contains labeling functions for ruling out Addison's Disease
