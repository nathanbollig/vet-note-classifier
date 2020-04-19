# -*- coding: utf-8 -*-
"""
Creates new columns after data cleaning from the HTML text.
New columns:
1. Cleaned_Text
2. History
3. Physical_Exam
4. Diagnoses
5. Diagnostic_Test
6. Treatment

@author: CHIT
"""

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup

search_words = {  # NB 3/17 - updated categories and key words
    'diagnoses': ['suspect', 'diagnosis', 'diagnose', 'problem list', 'problem'],
    'diagnostic_test': ['test', 'diagnostic', 'procedure'],
    'physical_exam': ['physical'],
    'history': ['history', 'concern'],
    'treatment': ['treatment', 'summary', 'to go home', 'comment', 'recommendation', 'instructions', 'plan', 'follow',
                  'medication']
}

# Lines needs to be removed from the text
removed_strings = [
    '[x if abnormal]',
    'please continue the same medication regimine for her future doctor visits to the hospital.',
    '<b>please contact us through phone 608-263-7600, or email at <b>primarycare@vetmed.wisc.edu</b> for any updates, questions, or concerns.</b>',
    'remember to either: <ul> <li>click the add rx button to insert medications dispensed today or <li>click the cpy comment button to insert the \
    current medication form for on-going medications</li></ul></font><!--  --></td></tr></tbody></table>',
    '[expand on above diagnoses as needed. do not simply reiterate. may include: summary of client discussion, differentials, and patient care.\
    *include alternative/future options for testing or treatment based on patient response.]',
    '<b>if you have a local veterinarian who referred you to uw veterinary care, we will send him/her a report for this visit.</b>',
    'please contact us via phone'
]


def process_data(train_path, output_path):
    # reading raw data
    data = pd.read_csv(train_path)
    data.species = data.species.str.strip()

    cols = data.columns
    cols = cols.insert(0, 'record_number')
    data['record_number'] = data.index
    data = data.loc[:, cols]

    data_canine = data.loc[data.species == 'Canine']
    new_cols = ["Cleaned_Text", "History", "Physical_Exam", "Diagnoses", "Diagnostic_Test",
                "Treatment"]  # NB 3/17 - update categories
    data_canine[new_cols] = data_canine.apply(lambda x: process_row(x.comment_text),
                                              axis=1, result_type="expand")
    data_canine.to_csv(output_path, index=False)


"""
NB 3/17 - added to limit character length of titles
"""


def filter_titles(titles):
    char_limit = 75

    for title in list(titles):
        keep = True
        if len(title) > char_limit:
            keep = False
        if keep == False:
            titles.remove(title)


def find_features_row(cleaned_text, titles, keywords):
    filter_titles(titles)  # NB 3/17

    lines = cleaned_text.split("\n")
    result_text = []
    for i, l in enumerate(lines):
        for keyword in keywords:
            if l.find(keyword) > -1 and (l.strip().lower() in titles):  # NB 3/17 - needed for match
                next_lines = ''
                if i + 1 < len(lines):
                    for j in range(1, len(lines) - i):
                        next_lines = lines[i + j]
                        if next_lines.strip().lower() in titles:  # NB 3/17 - needed for match
                            break
                        else:
                            result_text.append(next_lines)

    # NB 3/17 - Get unique lines in original order
    indexes = np.unique(result_text, return_index=True)[1]
    result_text = [result_text[index] for index in sorted(indexes)]

    result_text = " ".join(result_text)  # NB 3/17 - space to prevent words from being concatenated
    return data_cleaning(result_text)


def data_cleaning(text):
    # remove newline characters
    cleaned_text = re.sub('\\n', ' ', text)
    # remove optionals am/pm like text [WARNING: it will remove na(sodium), k (potassium)...etc]
    # cleaned_text = re.sub('[a-z]{1,}\/[a-z]{1,}', ' ', cleaned_text)

    # remove numbers and dates
    cleaned_text = re.sub('\[date\]|\[num\]', ' ', cleaned_text)
    # remove emails
    cleaned_text = re.sub('[a-z1-9]{1,}@[a-z]{1,}\.(com|edu|uk|co)', ' ', cleaned_text)
    # remove non a-z characters
    cleaned_text = re.sub('[^a-z\s]', ' ', cleaned_text)
    # removing all 1 or 2 char words
    cleaned_text = re.sub('\s[a-z]{1,2}\s', ' ', cleaned_text)
    cleaned_text = re.sub('\s[a-z]{1,2}\s', ' ', cleaned_text)
    # removing extract space between the words
    cleaned_text = re.sub('\s+', ' ', cleaned_text)

    return cleaned_text


def process_row(comment_html_o):
    if comment_html_o is np.nan:
        return '', '', '', '', '', ''

    # if verbose: print(comment_html)
    comment_html = comment_html_o.lower()
    comment_html = comment_html.replace("<b>", "<b>\n")
    comment_html = comment_html.replace("</b>", "\n</b>")
    comment_html = comment_html.replace("dr.", "dr")
    comment_html = re.sub('<div>\[\s\]\s.*?</div>', '', comment_html)

    for r_string in removed_strings:
        comment_html = comment_html.replace(r_string, '')
        # remove the entire line
        # TODO

    soup = BeautifulSoup(comment_html)

    titles = soup.findAll('b')
    titles = [t.text.strip().lower() for t in titles]

    soup_text = soup.text
    cleaned_text = re.sub('\[.*?\]', '', soup_text)
    cleaned_text = re.sub('\\xa0', '', cleaned_text)
    cleaned_text = re.sub('[ ]{2,}', '\n', cleaned_text)

    # removing all instruructive sentences like  [instruction]
    cleaned_text = re.sub('\[[a-z\s\d\-,:\(\)!;\.\"\\/#$\*%]{2,}\]', '\n', cleaned_text)
    # removing unchecked sentences
    cleaned_text = re.sub('\[\s\].*', '', cleaned_text)
    # remove [x] marks
    cleaned_text = re.sub('\[x\]', '', cleaned_text)

    # remove dates
    cleaned_text = re.sub('[0-9]{1,2}[\/,:][0-9]{1,2}[\/,:][0-9]{2,4}', '[DATE]', cleaned_text)

    # remove special characters
    cleaned_text = re.sub('[:,;%&*()$#-]', '', cleaned_text)

    # remove numbers
    cleaned_text = re.sub('\s[0-9\.].*?(\s|\n)', ' [NUM] ', cleaned_text)

    # remove remaining numbers
    cleaned_text = re.sub('\s[0-9\.].*?(\s|\n)', ' [NUM] ', cleaned_text)

    cleaned_text = cleaned_text.lower()

    history = find_features_row(cleaned_text, titles, search_words['history'])
    physical_exam = find_features_row(cleaned_text, titles, search_words['physical_exam'])
    diagnoses = find_features_row(cleaned_text, titles, search_words['diagnoses'])
    diagnostic_test = find_features_row(cleaned_text, titles,
                                        search_words['diagnostic_test'])  # NB 3/17 - redefined categories
    treatment = find_features_row(cleaned_text, titles, search_words['treatment'])

    cleaned_text = data_cleaning(cleaned_text)

    return cleaned_text, history, physical_exam, diagnoses, diagnostic_test, treatment


if __name__ == '__main__':
    process_data('../data/data_raw/data.csv',
                 '../data/data_processed/data_processed.csv')
