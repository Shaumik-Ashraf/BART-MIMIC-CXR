# bart_summarizer_find_tuned.py
# make sure you have transformers==4.5.0

import os;
import torch;
import csv;
import numpy as np;
import pandas as pd;

from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Trainer, TrainingArguments
from transformers.models.bart.modeling_bart import shift_tokens_right

ROOT = os.path.dirname(os.path.abspath(__file__));

os.execlp("pip", "install", "-r", os.path.join(ROOT, "transformers/seq2seq/requirements.txt"));

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");