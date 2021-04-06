# BART-MIMIC-CXR

Use BART word embedder to do NLP analysis on radiology reports from MIMIC-CXR v2.0.0

`run_summarization.py` is from huggingface team

## Dependencies
 - install pytorch
 - conda install -c huggingface transformers==3.3.1
 - conda install pandas
 - conda install nltk
 - pip install py-rouge

## Use
 - `mkdir data`
 - add mimic-cxr reports and study-lists data into data/
 - `python preprocessing`
 - ...


