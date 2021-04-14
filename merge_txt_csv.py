# -*- coding: utf-8 -*-
# merge_txt_csv.py
"""
merge txt file with csv files
CSV_FILE - conains subject_id, study_id, and actual columns
TXT_FILE - generated summaries per line
"""

import pandas as pd;
import os;

ROOT = os.path.dirname( os.path.abspath(__file__) );
CSV_FILE = "Summaries_Final_Baseline.csv"
TXT_FILE = "Outputs/truncated_generations.txt"
OUT_FILE = "Summaries_Final_Truncated.csv"

csv_path = os.path.join(ROOT, 'data', CSV_FILE);
txt_path = os.path.join(ROOT, 'data', TXT_FILE);
out_path = os.path.join(ROOT, 'data', OUT_FILE);

df = pd.read_csv(csv_path);
df.drop(columns=['prediction']);

f = open(txt_path, 'r');
text = f.read();
f.close();
l = text.split('\n');
#print(l);
s = pd.Series(data = l, name='prediction');

df['prediction'] = s;
df.to_csv(out_path, index=False);
