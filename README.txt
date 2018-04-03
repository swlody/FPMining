USAGE:
fpmine.py [-h] [--fp_growth] [--min_sup [MIN_SUP]] [--labels LABELS [LABELS ...]] [--count] filename

For example, for the adult.data dataset from http://archive.ics.uci.edu/ml/datasets/Adult:
python3 fpmine.py adult.data --min_sup 0.5 --labels age workclass fnlwgt education education-num marital-status occupation relationship race sex capital-gain capital-loss hours-per-week native-country

Both apriori.py and fp_growth.py take in a CSV file that contains a database of transactions and a minimum support. The default minimum support is 0.5, which means that a frequent itemset is one that occurs in at least 50% of the transactions. Different minimum supports can be specified with the --min-sup argument. The default transaction database is adults.data. The method that reads in the CSV assumes that the labels of each column are included in the 'labels' list. The length of this list also determines how many columns the method assumes exist in the CSV.