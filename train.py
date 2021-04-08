# train.py
!pip install "transformers==3.3.1"
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments
from transformers.modeling_bart import shift_tokens_right
import csv
import numpy as np

# Some code to load our dataset
REPORTS_DIR = os.path.join(ROOT, 'data', 'mimic-cxr-reports');
TRAIN_FILE = os.path.join(ROOT, 'data', 'train.csv');
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');

def load_data(file):
	with open(file) as fp:
		reader = csv.reader(fp, delimiter=' ')
		data_read = [row for row in reader]
		return data_read

#model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-large')
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

#ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
#inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='tf')

# Generate Summary
#summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
#print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])