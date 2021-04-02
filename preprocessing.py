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
import re

TEST_FRACTION = 0.3 # fraction for test set

ROOT = os.path.dirname( os.path.abspath(__file__) );
LIST_FILE = os.path.join(ROOT, 'data', 'cxr-study-list.csv.gz');
REPORTS_DIR = os.path.join(ROOT, 'data', 'mimic-cxr-reports');
TRAIN_FILE = os.path.join(ROOT, 'data', 'train.csv');
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');


print("================ Starting data preprocessing ==================");
print(f"Reading {os.path.basename(LIST_FILE)}...");
radiology_reports = pd.read_csv(LIST_FILE)['path']; # file paths as pandas series
train_len = int(len(radiology_reports) * (1.0 - TEST_FRACTION));
train_reports = radiology_reports[:train_len];
test_reports = radiology_reports[train_len:];
print("Done.");

print("Writing train.csv...");
f = open(TRAIN_FILE, 'w');
for report in train_reports:
	x = open(os.path.join(REPORTS_DIR, report))
	text = x.read().strip();
	re.sub("\n", "", text);
	data = re.split("(IMPRESSION:)|(IMPRESSIONS:)",text);
	print(data)
	if( (len(data)<2) or (data[1].strip() == "") ):
		print(f"Ommitting file {os.path.basename(report)} - no impressions section");
		continue; #toss out data and go to next textfile
	
	# do lematization here
	
	f.write(f"\"{data[0]}\",\"{data[1]}\"\n");
f.close();
print("Done.\n");
"""
print("Writing test.csv...");
f = File.open(TEST_FILE, 'w');
for report in test_reports:
	text = File.read(report);
	data = re.split("IMPRESSION:");
	if( (len(data)<2) or (data[1].strip().empty()) ):
		print(f"Ommitting file {os.path.basename(report)} - no impressions section");
		continue; #toss out data and go to next textfile
	
	# do lematization here
	
	f.write(f"\"{data[0]}\",\"{data[1]}\"\n");
f.close();
print("Done.\n");
"""
print("==================== End data preprocessing ======================");
