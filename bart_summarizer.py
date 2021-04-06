# summarizer.py
"""
Does abstractive summarization on MIMIC CXR Radiology reports, uses BART transformer

Requirements:
data/test.csv exists
data/summaries.csv can be created or overwritten
transformers==3.3.1
"""

import torch
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments
from transformers.modeling_bart import shift_tokens_right
import csv
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname( os.path.abspath(__file__) );
#TRAIN_FILE = os.path.join(ROOT, 'data', 'train.csv');
#VALIDATION_FILE = os.path.join(ROOT, 'data', 'validation.csv');
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');

LIMIT = 10; #set limit to -1 to do all data
SUMMARIES_FILE = os.path.join(ROOT, 'data', f"summaries_{LIMIT}.csv");

def load_file(filename):
	"""
	loads csv data and returns it as np matrix
	
	param: filename - path to csv file
	return: 2D numpy of csv data with text
	"""
	print(f"Loading data from {filename}...");
	df = pd.read_csv(filename)
	print(f"Done.");
	return( np.array(df) );

def load_bart(model_name='facebook/bart-large-cnn', tokenizer_name='facebook/bart-large'):
	"""
	loads pretrained BART model and tokenizer
	
	params: model_name - pretrained BART huggingface transformer download path, default: facebook/bart-large-cnn
		    tokenizer_name - pretrained BART huggingface tokenizer download path, default: facebook/bart-large
	return: (model, tokenizer)
	"""
	print(f"Loading pretrained model {model_name}...");
	model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
	print("Done.");
	print(f"Loading pretrained tokenizer {tokenizer_name}...");
	tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
	print("Done.");
	return((model, tokenizer));

def baseBart(article_to_summarize, model, tokenizer):
	"""
	runs BART summarization
	
	params: model - from load_bart()
		    tokenizer - from load_bart()
			article_to_summarize - text (string)
	return: generated abstractive summary (string)
	"""
	inputs = tokenizer([article_to_summarize], max_length=1024, return_tensors='pt')
	
	summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=25, early_stopping=True)
	return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

def write_csv_row(opened_file, row, model, tokenizer):
	"""
	generates abstractive summary and writes it to a file in csv format, 1 summary per row
	
	params: opened_file - open File object, actually any IO stream implementing write() works
		    row - a list/array containing [<subject id>, <study id>, <text to summarize>, <ground truth summary>]
			model - trained BART model
			tokenizer - BART tokenizer
	returns: generated summary
	"""
	comp_summary = baseBart(row[2], model, tokenizer)
	opened_file.write(f"\"{row[0]}\",\"{row[1]}\",\"{comp_summary}\",\"{row[3]}\"\n");
	return(comp_summary);

print("==================== Start abstractive summarization ======================");

data = load_file(TEST_FILE);
model, tokenizer = load_bart();

print(f"Writing {os.path.basename(SUMMARIES_FILE)}...");
f = open(SUMMARIES_FILE, 'w');
f.write(f"\"subject_id\",\"study_id\",\"prediction\",\"actual\"\n");
i = 0;
if LIMIT==-1: # based on the limit, print progress messages appropriately
	for row in data:
		write_csv_row(f, row, model, tokenizer);
		if( (i%1000 == 0) or (i+1 == LIMIT) ):
			print(f"Computed {i+1} summaries");
		i += 1;
elif LIMIT < 100:
	for row in data[:LIMIT]:
		write_csv_row(f, row, model, tokenizer);
		if( (i%(int(LIMIT/4)) == 0) or (i+1 == LIMIT)):
			print(f"Computed {i+1} summaries");
		i += 1;
else:
	for row in data[:LIMIT]:
		write_csv_row(f, row, model, tokenizer);
		if( (i%(int(LIMIT/8)) == 0) or (i+1 == LIMIT) ):
			print(f"Computed {i+1} summaries");
		i += 1;

f.close();
print("Done.\n");
print("==================== End abstractive summarization ======================");
