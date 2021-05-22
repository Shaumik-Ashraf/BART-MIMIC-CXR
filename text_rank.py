# text_rank.py
# run text rank for another baseline model against BART

import os;
import numpy as np;
import pandas as pd;
from textrank4zh import TextRank4Sentence;

ROOT = os.path.dirname(os.path.abspath(__file__));
TEST_FILE = os.path.join(ROOT, 'data', 'test.csv');
SUMMARIES_FILE = os.path.join(ROOT, 'data', 'extractive_summaries.csv');
LIMIT = -1;


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

def truncate(text, word_limit):
    words = text.strip().split();
    if( len(words) < word_limit ):
        return(text);
    
    first_fullstop = text.find('.');
    last_fullstop = text.rfind('.');
    if( first_fullstop==-1 or first_fullstop==last_fullstop ):
        text_idx = 0;
        for i, w in enumerate(words):
            text_idx += (len(w) + 1); #add 1 for space
            if( i >= word_limit ):
                break;
        return( text[:text_idx] );
    else:
        second_last_fullstop = text[:last_fullstop].rfind('.');
        return( truncate(text[:(second_last_fullstop+1)], word_limit) );        

def extractive_summarization(summarizer, text, word_limit = 25, err_tag="..."):
    #summarizer = TextRank4Sentence(); #model can be recycled
    summarizer.analyze(text=text, lower=True, source="all_filters");
    results = extractive_summarizer.get_key_sentences(num=1);
    if len(results) < 1:
        print(f"Failed to generate summary for {err_tag}")
        return("None.");
    
    return truncate(results[0]['sentence'], word_limit);

def write_csv_row(opened_file, row, model):
    summary = extractive_summarization(model, row[2], word_limit = 25, err_tag=row[1]);
    opened_file.write(f"\"{row[0]}\",\"{row[1]}\",\"{summary}\",\"{row[3]}\"\n");
    return(summary);

print("==================== Start extractive summarization ======================");

extractive_summarizer = TextRank4Sentence();
data = load_file(TEST_FILE);

print(f"Writing {os.path.basename(SUMMARIES_FILE)}...");
f = open(SUMMARIES_FILE, 'w');
f.write("\"subject_id\",\"study_id\",\"prediction\",\"actual\"\n");
i = 0;
if LIMIT==-1: # based on the limit, print progress messages appropriately
	for row in data:
		write_csv_row(f, row, extractive_summarizer);
		if( (i%1000 == 0) or (i+1 == LIMIT) ):
			print(f"Computed {i+1} summaries");
		i += 1;
elif LIMIT < 100:
	for row in data[:LIMIT]:
		write_csv_row(f, row, extractive_summarizer);
		if( (i%(int(LIMIT/4)) == 0) or (i+1 == LIMIT)):
			print(f"Computed {i+1} summaries");
		i += 1;
else:
	for row in data[:LIMIT]:
		write_csv_row(f, row, extractive_summarizer);
		if( (i%(int(LIMIT/8)) == 0) or (i+1 == LIMIT) ):
			print(f"Computed {i+1} summaries");
		i += 1;

f.close();
print("Done.\n");
print("==================== End extractive summarization ======================");
