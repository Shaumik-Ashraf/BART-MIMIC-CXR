# BART-MIMIC-CXR

Use BART word embedder to do NLP analysis on radiology reports from MIMIC-CXR v2.0.0

`run_summarization.py` is from huggingface team

## Install
 - install pytorch
 - conda install -c huggingface transformers==4.0.0
 - conda install pandas
 - conda install nltk

## Use
 - `mkdir data`
 - add mimic-cxr reports and study-lists data into data/
 - `python preprocessing`

## Todo
 - split train set into dev and validation sets
 - fix run_summarization_modified.py code
 - confirm if baseline.py works
 - iterate baseline.py over entire dataset and compute ROGUE score

