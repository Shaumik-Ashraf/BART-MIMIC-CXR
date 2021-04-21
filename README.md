# BART-MIMIC-CXR

We use BART word embedder to do the abstractive summarization NLP task on radiology reports from MIMIC-CXR v2.0.0.

## Use:
 - clone this repository
 - in this repository, `mkdir data`
 - from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) download cxr-study-list.csv.gz and place it in the data folder
 - also from MIMIC-CXR, download mimic-cxr-reports.zip, unzip it, and place it in the data folder
 - run `python preprocessing`, which generates train.csv, validate.csv, and test.csv in the data folder
 - use the `notebook/*.ipynb` files to run the various BART models
   + the baseline model creates a summaries.csv file in data/ that contains both ground-truth and generated summaries
   + the fine-tuned and fune-tuned with truncation models create a `.txt` file, that must be converted to a `.csv` file by running `python merge_txt_csv.py`
 - run `python rouge_score` or `python word_count` to get the ROUGE scores and word count of the summaries respectively. 

Note: You may need to go into the python file to set the proper path or filename, which is all configurable from variables at the top of the file.
