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
f.write(f"\"subject_id\",\"study_id\",\"findings\",\"impression\"\n");
ommitted = 0;
progress = 1;
for report in train_reports:
	x = open(os.path.join(REPORTS_DIR, report));
	text = x.read().strip();
	text = re.sub("\n", "", text);
	text = re.sub(",", "", text); # causes issues with CSV
	idx = text.find("NOTIFICATION");
	if( idx > 0 ):
		text = text[:idx];
	data = re.split(r'impression.?(?::|" ")',text, flags=re.IGNORECASE);
	data = [s.strip() for s in data]
	if (progress % 10000 == 0):
		print(f'Read {progress} files so far...');
	progress += 1;
	if( (len(data)<2) or (data[1].strip() == "") ):
		ommitted += 1;
		continue; #toss out data and go to next textfile
	folders = report.split('/');
	f.write(f"\"{folders[2]}\",\"{folders[3].split('.')[0]}\",\"{data[0]}\",\"{data[1]}\"\n");
f.close();
print(f"Ommited {ommitted} files out of {progress} total files in train.\n")
print("Done.\n");
print("Writing test.csv...");
f = open(TEST_FILE, 'w');
f.write(f"\"subject_id\",\"study_id\",\"findings\",\"impression\"\n");
ommitted = 0;
progress = 1;
for report in test_reports:
	x = open(os.path.join(REPORTS_DIR, report));
	text = x.read().strip();
	text = re.sub("\n", "", text);
	text = re.sub(",", "", text); # causes issues with CSV
	idx = text.find("NOTIFICATION");
	if( idx > 0 ):
		text = text[:idx];
	data = re.split(r'impression.?(?::|" ")',text, flags=re.IGNORECASE);
	data = [s.strip() for s in data]
	if (progress % 10000 == 0):
		print(f'Read {progress} files so far...');
	progress += 1;
	if( (len(data)<2) or (data[1].strip() == "") ):
		ommitted += 1;
		continue; #toss out data and go to next textfile
	folders = report.split('/');
	f.write(f"\"{folders[2]}\",\"{folders[3].split('.')[0]}\",\"{data[0]}\",\"{data[1]}\"\n");
f.close();
print(f"Ommited {ommitted} files out of {progress} total files in test.\n")
print("Done.\n");
print("==================== End data preprocessing ======================");
