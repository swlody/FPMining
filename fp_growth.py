import csv
import argparse

labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'probability']

def get_transactions_from_csv(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        database = []
        for row in reader:
            database.append([(labels[i], row[i].strip()) for i in range(len(labels) - 2)]) # Not including probability
        return database

def frequent_items(transactions):
    F = []
    supports = []
    return F, supports

class fp_node:
    def __init__(self, value, parent):
        self.value = value
        self.parent = parent
        self.children = set()

def gen_freq_1_itemsets(D, min_sup):
    headers = dict()
    for t in D:
        for item in t:
            headers[item] = headers.get(item, 0) + 1
    # print(headers)
    return set({k: v for k, v in headers.items() if v >= min_sup}.keys()), headers

def gen_fp_tree(D, min_sup):
    freq_items, headers = gen_freq_1_itemsets(D, min_sup)
    print(freq_items)

def fp_growth(fp_tree, sup_thresh):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FP-Growth Algorithm on a given transaction database.")
    parser.add_argument('file', nargs='?', default='adult.data', help="Path to CSV file with transaction database.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.5, help="Minimum support an itemset must meet to avoid being pruned.")
    args = parser.parse_args()

    transactions = get_transactions_from_csv(args.file)
    tree = gen_fp_tree(transactions, args.min_sup * len(transactions))
    fp_growth(tree, args.min_sup * len(transactions))
