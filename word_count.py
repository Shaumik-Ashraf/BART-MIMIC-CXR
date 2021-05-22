# -*- coding: utf-8 -*-
# word_count.py
# gets wordcount of summaries columns and basic statistics


import pandas as pd;
import numpy as np;
import os;

ROOT = os.path.dirname( os.path.abspath(__file__) );
SUMMARY_FILE_NAME = "extractive_summaries.csv"
COMPUTE_FOR_TRUTH = False; #only possible with CSV format

def word_count(s):
    """
    naive word count (also counts symbols, punctuation, etc., but good for our purposes)
    """
    return( len(s.strip().split()) );


summary_path = os.path.join(ROOT, 'data', SUMMARY_FILE_NAME);

if SUMMARY_FILE_NAME.endswith(".csv"):
    df = pd.read_csv(summary_path);
    num_summaries = len(df);
    generated_summaries = df['prediction'];
    if COMPUTE_FOR_TRUTH:
        true_summaries = df['actual'];
    del df;
else: #txt format
    f = open(summary_path,'r');
    x = f.readlines(); #one summary per line
    f.close();
    generated_summaries = pd.Series(data=x, name='prediction');

print(f"For {SUMMARY_FILE_NAME}:");

prediction_word_counts = generated_summaries.apply(word_count);
print(prediction_word_counts.describe(), "\n");


if COMPUTE_FOR_TRUTH:
    actual_word_counts = true_summaries.apply(word_count);
    print(actual_word_counts.describe(), "\n");
