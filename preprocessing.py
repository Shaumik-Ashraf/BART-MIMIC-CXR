# preprocessing.py
"""
parses MIMIC-CXR radiology reports from data/mimic-cxr-reports/ into data/train.csv and data/test.csv

train.csv contains two columns with each column wrapped in double quotes; the first column contains
the input text (radiology examination, technique, comparison, and findings) while the second
column contains the output text (impressions). All reports without the term "IMPRESSIONS:" are ommitted

test.csv has the same structure as train.csv. 

The processing also lematizes all of the terms using nltk and strips whitespace.

REQUIREMENTS:
 - data/mimic-cxr-reports/*
 - data/cxr-study-list.csv.gz
 - overwrite data/train.csv
 - overwrite data/test.csv
"""

import os;
import pandas as pd;

TEST_FRACTION = 0.3 # fraction for test set

ROOT = os.dirname( os.abspath(__file__) );
LIST_FILE = os.path.join(ROOT, 'data', 'cxr-study-list.csv.gz');
REPORTS_DIR = os.path.join(ROOT, 'data', 'mimic-cxr-reports');
TRAIN_FILE = os.path.join(ROOT, 'data', 'train.csv');
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');


text_files = pd.read_csv(LIST_FILE)['path']; # file paths as pandas series
train_len = len(text_files) * (1.0 - TEST_FRACTION);
train_files = text_files[:train_len];
test_files = text_files[train_len:];


f = File.open(TRAIN_FILE, 'w');
for filename in train_files:
	text = File.read(filename);
	data = text.split("IMPRESSIONS:");
	if( (len(data)<2) or (data[1].strip().empty()) ):
		continue; #toss out data and go to next textfile
	
	# do lematization here
	
	f.write(f"\"{data[0]}\"");
	
f.close();
	