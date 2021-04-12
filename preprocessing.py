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

TEST_FRACTION = 0.1 # fraction for test set
VALIDATION_FRACTION = 0.1

ROOT = os.path.dirname( os.path.abspath(__file__) );
LIST_FILE = os.path.join(ROOT, 'data', 'cxr-study-list.csv.gz');
REPORTS_DIR = os.path.join(ROOT, 'data', 'mimic-cxr-reports');
TRAIN_FILE = os.path.join(ROOT, 'data', 'train.csv');
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');
VALIDATION_FILE = os.path.join(ROOT, 'data', 'validation.csv');



def remove_notification_section(text):
	"""
	We noticed that some reports have a notification section after
	the impressions (summary) section, which was impeding our data, so
	we decided to remove this section all together. We use various rule-based
	mechanisms to parse and remove the notification section.

	params: text
	returns: text with notification section removed
	"""
	idx = text.rfind("NOTIFICATION");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("telephone notification");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("Telephone notification");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("These findings were");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("Findings discussed");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("Findings were");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("This preliminary report");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("Reviewed with");
	if( idx > 0 ):
		text = text[:idx];
	idx = text.rfind("A preliminary read");
	if( idx > 0 ):
		text = text[:idx];
	return(text);

def sanitize(text):
	"""
	Cleanses the text to be written in CSV, which will be fed directly to
	the summarizer. Tokenization and lemmatization is not performed in this
	step, as the summarizer performs those directly.

	params: text
	returns: cleaned text
	"""
	text = text.strip();
	text = re.sub("\n", "", text);
	text = re.sub(",", "", text);
	# Remove all text before FINDINGS: section
	regex = r'^(.*finding.?:)'

	if( re.search(regex, text, flags=re.IGNORECASE)==None ): #if no summary
		return None;

	text = re.sub(regex,"", text, flags=re.IGNORECASE);
	text = remove_notification_section(text);
	return(text);

def split(slicable, fraction):
	"""
	splits data into test-train set or dev-validation set; does not shuffle.

	params: slicable - an object that responds to len() and [], works on dataframes
	        fraction - a value between 0 and 1
	returns: (x, y) - where x has (1-fraction) percent entries and y has the rest
	"""
	partition = int(len(slicable) * (1.0 - fraction));
	return( (slicable[:partition], slicable[partition:]) );

def parse_summary(text):
	"""
	parses and separates input text from summary in cxr reports, returns None if
	not found

	params: text
	returns: None or [input_text, summary]
	"""

	regex = r'impression.?(?::|" ")'

	if( re.search(regex, text, flags=re.IGNORECASE)==None ): #if no summary
		return None;

	data = re.split(regex, text, flags=re.IGNORECASE);
	data[0] = data[0].strip();
	data[1] = data[1].strip();

	return(data);

def write_csv(filename, reports):
	"""
	writes a csv file for summarization. The CSV file has four columns: "subject_id",
	"study_id", "findings", and "impression" based on MIMIC-CXR reports. "findings"
	contains the input text, and "impression" contains the true summary.

	params: filename - name of csv file to write, will overwrite if it exists
		reports - dataframe of cxr reports from cxr-study-list file
	"""
	print(f"Writing {filename}...");
	f = open(filename, 'w');
	f.write(f"\"subject_id\",\"study_id\",\"findings\",\"impression\"\n");
	ommitted = 0;
	progress = 1;
	for report in reports:
		x = open(os.path.join(REPORTS_DIR, report));
		text = x.read();
		x.close();
		text = sanitize(text);
		if( text==None ):
			ommitted += 1;
			continue; #toss out data and go to next textfile

		if (progress % 10000 == 0):
			print(f'Read {progress} files so far...');
		progress += 1;

		data = parse_summary(text);
		if( (data==None) or (data[0]=='') or (data[1]=='') ):
			ommitted += 1;
			continue; #toss out data and go to next textfile

		folders = report.split('/');
		f.write(f"\"{folders[2]}\",\"{folders[3].split('.')[0]}\",\"{data[0]}\",\"{data[1]}\"\n");
	f.close();
	print(f"Ommited {ommitted} files out of {progress} total files in dataset.\n")
	print("Done.\n");


print("================ Starting data preprocessing ==================");

print(f"Reading {os.path.basename(LIST_FILE)}...");
radiology_reports = pd.read_csv(LIST_FILE)['path']; # file paths as pandas series
train_reports, test_reports = split(radiology_reports, TEST_FRACTION);
print("Done.");

# if you want validation set:
train_reports, validation_reports = split(train_reports, VALIDATION_FRACTION / (1 - TEST_FRACTION));
write_csv(VALIDATION_FILE, validation_reports);

# sanity check
#print(train_reports);
#print(validation_reports);
#print(test_reports);

write_csv(TRAIN_FILE, train_reports);
write_csv(TEST_FILE, test_reports);

print("==================== End data preprocessing ======================");
