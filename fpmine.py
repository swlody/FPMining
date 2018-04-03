from __future__ import print_function
from argparse import ArgumentParser
from itertools import combinations
import csv


def main():
    parser = init_parser()
    args = parser.parse_args()

    database = csv_to_transaction_db(args.filename, args.labels)

    if args.fp_growth:
        freq_itemsets = fp_growth(database, args.min_sup)
    else:
        freq_itemsets = apriori(database, args.min_sup)

    for j in range(1, len(freq_itemsets)):
        if args.count:
            print("Frequent", j, "itemsets:", len(freq_itemsets[j]) if freq_itemsets[j] else 0)
        else:
            print("Frequent", j, "itemsets:", freq_itemsets[j])
            if j != len(freq_itemsets) - 1:
                print()


def csv_to_transaction_db(filename, labels):
    """Produce a list of transactions, where each transaction is a set of tuples
    containing the label and value of the item. In theory, this should be the only
    method that needs to be reimplemented for simple datasets."""
    with open(filename) as file:
        csvreader = csv.reader(file, skipinitialspace=True)
        # Check to see if the CSV has a header by looking at the first few lines
        has_header = csv.Sniffer().has_header(''.join([file.readline(), file.readline(), file.readline()]))
        if not labels:
            if has_header:
                # If it does, we read in the labels from the first line of the CSV
                file.seek(0)
                labels = csvreader.__next__()
            else:
                # If it doesn't, print an error and exit
                exit("It appears that the CSV file does not have a header. "
                     "Please specify the labels of each column with --labels.")
        else:
            # Check if more labels were specified than there are rows in the CSV
            if len(labels) > len(file.readline()):
                exit("More labels were specified than there are rows in the CSV.")
            # and seek back to the first row of the file unless there's a header row
            if not has_header:
                file.seek(0)
        database = []
        # Ignore the probability label - I don't know what it's used for
        for row in csvreader:
            # Create a tuple of the label and the value of each item in the row
            # this will ensure different hashes in the case of values
            # from different columns coincidentally being equal.
            # Each transaction in the database is a set of such tuples.
            transaction = {(labels[i], row[i]) for i in range(len(labels))}
            database.append(transaction)
        return database


def join(l1, l2):
    """Join two itemsets that have all items equal except the last"""
    copy = list(l1)
    copy.append(l2[-1])
    return copy


def gen_freq_1_itemsets(database, min_sup):
    """Get all items in the database that meet the minimum support"""
    count = {}
    for transaction in database:
        for item in transaction:
            # Use default of 0 for count.get() if item is not already counted
            count[item] = count.get(item, 0) + 1
    return [[k] for k, v in count.items() if v >= min_sup]


def has_infrequent_subset(candidate, itemsets, k):
    """Returns true if any (k-1)-subset of the candidate itemset is not an L_(k-1) frequent itemset"""
    return all(list(subset) not in itemsets for subset in combinations(candidate, k-1))


def candidates(itemsets, k):
    """Apriori method to generate candidate itemsets"""
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
    """
    Run the Apriori algorithm on the database with the given minimum support

    @param: database A list of transactions, where each transaction is a list or set of items
    @param: min_sup The minimum support for an itemset to be considered frequent. Must be a number between 0 and 1.

    @returns: itemsets_list A list of frequent k-itemsets indexable with itemsets_list[k]
    """
    
    # Convert from relative (ratio of occurrences) to absolute (number of occurrences) support
    min_sup *= len(database)
    itemsets_list = [None, gen_freq_1_itemsets(database, min_sup)]
    k = 2
    while itemsets_list[k-1]:
        itemsets_list.append([])
        # For each potential candidate
        for candidate in candidates(itemsets_list[k-1], k):
            count = 0
            # Count the number of times it appears in the database
            for transaction in database:
                if all(item in transaction for item in candidate):
                    count += 1
            # If candidate meets the minimum support, add it to the list of frequent-k itemsets
            if count >= min_sup:
                itemsets_list[k].append(candidate)
        k += 1
    return itemsets_list


class fp_tree:
    pass


class fp_node:
    pass


def fp_growth(database, min_sup):
    pass


def init_parser():
    """Initialize the argument parser"""
    parser = ArgumentParser(description="Run Apriori Algorithm on a given transaction database.")
    parser.add_argument('filename',
                        help="Path to CSV file with transaction database.")
    parser.add_argument('--fp_growth', action='store_true',
                        "Use the FP-Growth algorithm for frequent pattern mining rather than Apriori.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.2,
                        help="Minimum support an itemset must meet to avoid being pruned. "
                             "Must be between 0 and 1. (default 0.2)")
    parser.add_argument('--labels', nargs='+',
                        help="List of labels for each column in the CSV. Only the first n columns of the CSV will be "
                             "read if n labels are specified and an error will be thrown if more labels are specified "
                             "than there are columns in the CSV. If no labels are specified, it is assumed that the "
                             " CSV contains a header with the labels of each column.")
    parser.add_argument('--count', action='store_true',
                        help="Just display the counts of each k-itemset, rather than printing the entire itemset.")
    return parser


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
