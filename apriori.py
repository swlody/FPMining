from __future__ import print_function
from argparse import ArgumentParser
from itertools import combinations
from csv import reader


def main():
    parser = init_parser()
    args = parser.parse_args()

    database = csv_to_transaction_db(args.filename)
    freq_itemsets = apriori(database, args.min_sup)

    for j in range(len(freq_itemsets)):
        # print("Frequent", j + 1, "itemsets:", len(freq_itemsets[j]) if freq_itemsets[j] else 0)
        print("Frequent", j + 1, "itemsets:", freq_itemsets[j])
        if j != len(freq_itemsets) - 1:
            print()


def csv_to_transaction_db(filename):
    """ Produce a list of transactions, where each transaction is a set of tuples
    containing the label and value of the item. In theory, this should be the only
    method that needs to be reimplemented for simple datasets. """
    with open(filename) as file:
        database = []
        # Ignore the probability label - I don't know what it's used for
        labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        for row in reader(file, delimiter=','):
            # Create a tuple of the label and the value of each item in the row
            # this will ensure different hashes in the case of values
            # from different columns coincidentally being equal.
            # Each transaction in the database is a set of such tuples.
            transaction = {(labels[i], row[i].strip()) for i in range(len(labels))}
            database.append(transaction)
        return database


def join(l1, l2):
    """ Join two itemsets that have all items equal except the last """
    copy = list(l1)
    copy.append(l2[-1])
    return copy


def gen_freq_1_itemsets(database, min_sup):
    """ Get all items in the database that meet the minimum support """
    count = {}
    for transaction in database:
        for item in transaction:
            # Use default of 0 for count.get() if item is not already counted
            count[item] = count.get(item, 0) + 1
    return [[k] for k, v in count.items() if v >= min_sup]


def has_infrequent_subset(candidate, itemsets, k):
    """ Returns true if any (k-1)-subset of the candidate itemset is not an L_(k-1) frequent itemset """
    return all(list(subset) not in itemsets for subset in combinations(candidate, k-1))


def candidates(itemsets, k):
    """ Apriori method to generate candidate itemsets """
    length = k - 2
    for l1 in itemsets:
        for l2 in itemsets:
            # If the itemsets are the same except for the last item
            if all(l1[i] == l2[i] for i in range(length)) and l1[length] < l2[length]:
                # Conveniently (and necessary for has_infrequent_subset() to work correctly),
                # the candidate list will always be sorted
                candidate = join(l1, l2)
                if not has_infrequent_subset(candidate, itemsets, k):
                    yield candidate


def apriori(database, min_sup):
    # Convert from relative (ratio of occurrences) to absolute (number of occurrences) support
    min_sup *= len(database)
    itemsets_list = [gen_freq_1_itemsets(database, min_sup)]
    k = 2
    while itemsets_list[k-2]:
        itemsets_list.append([])
        # For each potential candidate
        for candidate in candidates(itemsets_list[k-2], k):
            count = 0
            # Count the number of times it appears in the database
            for transaction in database:
                if all(item in transaction for item in candidate):
                    count += 1
            # If candidate meets the minimum support, add it to the list of frequent-k itemsets
            if count >= min_sup:
                itemsets_list[k-1].append(candidate)
        k += 1
    return itemsets_list


def init_parser():
    parser = ArgumentParser(description="Run Apriori Algorithm on a given transaction database.")
    parser.add_argument('filename', nargs='?', default='adult.data', help="Path to CSV file with transaction database.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.5,
                        help="Minimum support an itemset must meet to avoid being pruned.")
    return parser


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
