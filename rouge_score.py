# ROUGE implementation from: https://github.com/Diego999/py-rouge

import rouge;
import pandas as pd;
import os;

ROOT = os.path.dirname( os.path.abspath(__file__) );
SUMMARY_FILE_NAME = "Base_Bart_Summary_100.csv"

scorer = rouge.Rouge(metrics=['rouge-l'],
		     max_n=4,
		     limit_length=True,
		     length_limit=30,
                     length_limit_type='words',
		     apply_avg=True,
		     apply_best=False,
                     alpha=0.5,
		     weight_factor=1.0,
		     stemming=True)

""" Example:
generated = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
truth = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

score = scorer.get_scores([generated], [truth]);
print(score);

# here, score contains: {'rogue-l': {'p': x, 'r':, y, 'f': z}}
#	where x, y, z are precision, recall, and f1-scores respectively
"""

summary_path = os.path.join(ROOT, 'data', SUMMARY_FILE_NAME);
df = pd.read_csv(summary_path);
num_summaries = len(df);
generated_summaries = df['prediction'];
true_summaries = df['actual'];
del df;

print("Calculating scores...");
scores = scorer.get_scores(generated_summaries, true_summaries);
print("Done.");

print(scores);