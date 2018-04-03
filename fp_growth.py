from __future__ import print_function
from argparse import ArgumentParser
from csv import reader


def main():
    parser = init_parser()
    args = parser.parse_args()

    database = csv_to_transaction_db(args.filename)
    tree = gen_fp_tree(database, args.min_sup)
    fp_growth(tree)


class fp_node:
    def __init__(self, value, parent):
        self.value = value
        self.parent = parent
        self.children = set()


class fp_tree:
    def __init__(self):
        pass


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


def frequent_items(transactions):
    F = []
    supports = []
    return F, supports


def gen_fp_tree(D, min_sup):
    freq_items, headers = gen_freq_1_itemsets(D, min_sup)
    print(freq_items)


def fp_growth(fp_tree, sup_thresh):
    pass


def init_parser():
    parser = ArgumentParser(description="Run FP-Growth Algorithm on a given transaction database.")
    parser.add_argument('filename', nargs='?', default='adult.data', help="Path to CSV file with transaction database.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.5,
                        help="Minimum support an itemset must meet to avoid being pruned.")
    return parser


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
