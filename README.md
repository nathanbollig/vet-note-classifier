# vet-note-classifier
Applying weak supervision to train a classifier of veterinary clinical notes

## Data Columns
Load data using the `load()` method in the functions.py module. The resulting columns are as follows.
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
