from itertools import combinations
import csv
import argparse

labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'probability']

def gen_transactions_from_csv(file):
    with open(file, newline='') as csvfile:
        database = []
        for row in csv.reader(csvfile, delimiter=','):
            # Create a tuple of the label and the value of each item in the row
            # this will ensure different hashes in the case of values
            # from different columns coincidentally being equal
            database.append([(labels[i], row[i].strip()) for i in range(len(labels) - 1)])
            # length - 1 to ignore probability
        return database

def join(l1, l2):
    """ Join two itemsets that have all items equal except the last """
    copy = list(l1)
    copy.append(l2[-1])
    return copy

def gen_freq_1_itemsets(database, min_sup):
    """ Get all items in the database that meet the minimum support """
    count = dict()
    for transaction in database:
        for item in transaction:
            count[item] = count.get(item, 0) + 1
    return [[k] for k, v in count.items() if v >= min_sup]

def has_infrequent_subset(c, L, k):
    return all(list(s) not in L for s in combinations(c, k-1))

def gen_candidates(L, k):
    """ Apriori method to generate candidate itemsets """
    length = k-2
    for l1 in L:
        for l2 in L:
            if all(l1[i] == l2[i] for i in range(length)) and l1[length] < l2[length]:
                c = join(l1, l2)
                if not has_infrequent_subset(c, L, k):
                    yield c

def apriori(D, min_sup):
    # Convert from relative to absolute support
    min_sup *= len(D)
    L = [gen_freq_1_itemsets(D, min_sup)]
    k = 2
    while L[k-2]:
        L.append([])
        # For each potential candidate
        for c in gen_candidates(L[k-2], k):
            count = 0
            # Count the number of times it appears in the database
            for t in D:
                if all(item in t for item in c):
                    count += 1
            # If candidate meets the minimum support, add it to the list of frequent-k itemsets
            if count >= min_sup:
                L[k-1].append(c)
        k += 1
    return L

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Apriori Algorithm on a given transaction database.")
    parser.add_argument('file', nargs='?', default='adult.data', help="Path to CSV file with transaction database.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.5, help="Minimum support an itemset must meet to avoid being pruned.")
    args = parser.parse_args()

    transactions = gen_transactions_from_csv(args.file)
    freq_itemsets = apriori(transactions, args.min_sup)

    for i in range(len(freq_itemsets)):
        print("Frequent", i+1 ,"itemset:", len(freq_itemsets[i]) if freq_itemsets[i] else 0)
        print()
