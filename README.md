# BART-MIMIC-CXR

Use BART word embedder to do NLP analysis on radiology reports from MIMIC-CXR v2.0.0

`run_summarization.py` is from huggingface team

## Dependencies
 - have pytorch installed
 - `conda activate <env with pytorch>`
 - `pip install transformers==3.3.1`
 - `pip install pandas`
 - `pip install nltk`
 - `pip install py-rouge`

## Use
 - clone and cd into repository
 - `mkdir data`
 - add mimic-cxr-reports and study-lists data into data/
 - `python preprocessing`
 - `python bart_summarizer`
 - `python rouge_score`


