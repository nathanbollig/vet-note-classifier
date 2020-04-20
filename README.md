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

## Instructions for Manual Classification
Note: At this time, I suspect prevalance of positive labels in the dataset could be as low as 5%, but we will get a better sense of this as we proceed with labeling.

The following articles provide the background for the labeling criteria summarized below. 
 * S. C. Klein and M. E. Peterson, “Canine hypoadrenocorticism: Part I,” Can Vet J, vol. 51, no. 1, pp. 63–69, Jan. 2010. (
"Clinical signs and physical examination findings" and "Laboratory abnormalities")
 * K. Van Lanen and A. Sande, “Canine hypoadrenocorticism: pathogenesis, diagnosis, and treatment,” Top Companion Anim Med, vol. 29, no. 4, pp. 88–95, Dec. 2014, doi: 10.1053/j.tcam.2014.10.001. ("Clinical findings" and "Laboratory findings")

The easiest way to confirm a positive label is the presence of post-hoc evidence of clinical suspicion of Addison's Disease. This would be one of:
* Patient has a diagnosis or problem of "hypoadrenocorticism" or "Addison's Disease" mentioned in the note. A positive label should be assigned even if the note indicates suspicion without a definitive diagnosis.
* Patient was given a "baseline cortisol" or "ACTH stimulation" ("ACTH stim") test, regardless of its outcome.
* If the patient has been, is, or will be prescribed one of the following drugs, they likely have a diagnosis of Addison's Disease:
  * Desoxycorticosterone pivalate (DOCP)
  * Fludrocortisone acetate (florinef)
  
In other cases, a clinician should be suspecting the disease if the following conditions are met:
* Patient has **signs of GI or nonspecific disease, or hypovolemic shock**, indicated by any of the following:
  * Inappetance
  * Anorexia
  * Vomiting or regurgitation
  * Diarrhea
  * Melana and hematochezia
  * Abdominal pain
  * Weight loss
  * Lethargy
  * Depression
  * Weakness
  * Shaking
  * Hair loss
  * Hypovolemic shock: bradycardia or tachycardia, collapse, hypothermia, weak pulses, or poor capillary refill time
* The patient has **not** been diagnosed with one of these diseases:
  * Kidney failure (acute kidney injury, AKI, acute kidney disease, AKD)
  * GI parasites (worms)
  * Primary liver disease (not just evidence of liver damage on chemistry panel)
  * Acute pancreatitis
* The patient does **not** have a history of toxin ingested noted.
* In most but not all cases, the patient will have some abnormal results on a screening chemistry panel or CBC. If the other conditions are met but none of the following are present (or the patient has an opposing or contradictory result to these), this warrants further review. These are loosely in order from more common to less common findings.
  * Low sodium (Na), i.e. hyponatremia
  * High potassium (K), i.e. hyperkalemia
  * Low chloride (Cl), i.e. hypochloremia
  * Azotemia: increased BUN and/or creatinine
  * Mild to moderate normocytic, normochromic, nonregenerative anemia (PCV 20-35%)
  * High calcium, i.e. hypercalcemia
  * Absence of a stress leukogram in a sick animal (normal CBC is considered abnormal here)
  * High lymphocytes, i.e. lymphocytosis
  * Mild acidosis indicated by low bicarbonate (HCO3) or low pH
  * Low blood glucose, i.e. hypoglycemia
  * Low albumin, i.e. hypoalbuminemia
  * Low cholesterol, i.e. hypocholesterolemia
