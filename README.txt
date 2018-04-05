USAGE:
fpmine.py [-h] [--fp_growth] [--min_sup [MIN_SUP]] [--labels LABELS [LABELS ...]] [--count] filename



For example, for the adult.data dataset from http://archive.ics.uci.edu/ml/datasets/Adult:

python3 fpmine.py adult.data --min_sup 0.5 --labels age workclass fnlwgt education education-num marital-status occupation relationship race sex capital-gain capital-loss hours-per-week native-country



Or just add a header to the csv:

age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, probability

and run with:

python3 fpmine.py adult.data --min_sup 0.5



Both apriori.py and fp_growth.py take in a CSV file that contains a database of transactions and a minimum support. The default minimum support is 0.5, which means that a frequent itemset is one that occurs in at least 50% of the transactions. Different minimum supports can be specified with the --min-sup argument. The default transaction database is adults.data. The method that reads in the CSV assumes that the labels of each column are included in the 'labels' list. The length of this list also determines how many columns the method assumes exist in the CSV.


[Item(label='capital-loss', value='0'), Item(label='workclass', value='Private')]
[Item(label='capital-loss', value='0'), Item(label='sex', value='Male')]
[Item(label='race', value='White'), Item(label='workclass', value='Private')]
[Item(label='race', value='White'), Item(label='sex', value='Male')]
[Item(label='native-country', value='United-States'), Item(label='race', value='White')]
[Item(label='native-country', value='United-States'), Item(label='workclass', value='Private')]
[Item(label='native-country', value='United-States'), Item(label='sex', value='Male')]
[Item(label='capital-gain', value='0'), Item(label='workclass', value='Private')]
[Item(label='capital-gain', value='0'), Item(label='sex', value='Male')]