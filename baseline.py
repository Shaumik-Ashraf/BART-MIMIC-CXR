# baseline.py
# runs basic summarization using transformers pipeline on train

import pandas as pd;
from transformers import pipeline;

dataset = pd.read_csv("data/train.csv");
summarizer = pipeline("summarization");

text1 = dataset["findings"][0];
print("Text: ", text1, "\n");
summary = summarizer(text1, min_length=5, max_length=30);
print("Abstractive Summary: ", text1, "\n");
print("Ground Truth Summary: ", dataset["impression"][0]);

