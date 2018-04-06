from __future__ import print_function
from argparse import ArgumentParser
from itertools import combinations, groupby
from collections import namedtuple
import csv

# FP-Growth does not work yet!!!
def init_parser():
    """Initialize the argument parser"""
    parser = ArgumentParser(description="Run Apriori Algorithm on a given transaction database.")
    parser.add_argument('filename',
                        help="Path to CSV file with transaction database.")
    parser.add_argument('--fp_growth', action='store_true',
                        help="Use the FP-Growth algorithm for frequent pattern mining rather than Apriori.")
    parser.add_argument('-s', '--min_sup', type=float, nargs='?', default=0.2,
                        help="Minimum support an itemset must meet to avoid being pruned. "
                             "Must be between 0 and 1. (default 0.2)")
    parser.add_argument('-l', '--labels', nargs='+',
                        help="List of labels for each column in the CSV. Only the first n columns of the CSV will be "
                             "read if n labels are specified and an error will be thrown if more labels are specified "
                             "than there are columns in the CSV. If no labels are specified, it is assumed that the "
                             "CSV contains a header with the labels of each column.")
    parser.add_argument('-c', '--count', action='store_true',
                        help="Just display the counts of each k-itemset, rather than printing the entire itemset.")
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    database = csv_to_transaction_db(args.filename, args.labels)

    if args.fp_growth:
        freq_itemsets = fp_growth(database, args.min_sup*len(database))
    else:
        freq_itemsets = apriori(database, args.min_sup*len(database))

    for j in range(1, len(freq_itemsets)-1):
        if args.count:
            print("Frequent", j, "itemsets:", len(freq_itemsets[j]) if freq_itemsets[j] else 0)
        else:
            print("Frequent", j, "itemsets:")
            for itemset in freq_itemsets[j]:
                print(itemset)
            if j != len(freq_itemsets) - 2:
                print()


def csv_to_transaction_db(filename, labels):
    """Produce a list of transactions, where each transaction is
    a set of tuples containing the label and value of the item"""
    with open(filename) as file:
        csvreader = csv.reader(file, skipinitialspace=True)
        # Check to see if the CSV has a header by looking at the first few lines
        has_header = csv.Sniffer().has_header(''.join([file.readline(), file.readline(), file.readline()]))
        if not labels:
            if has_header:
                # If it does, we read in the labels from the first line of the CSV
                file.seek(0)
                labels = csvreader.next()
            else:
                # If it doesn't, print an error and exit
                exit("Error: it appears that the CSV file does not have a header row. "
                     "Please specify the labels of each column with --labels or add a header row to the CSV.")
        else:
            # Check if more labels were specified than there are rows in the CSV
            if len(labels) > len(file.readline()):
                exit("Error: more labels were specified than there are rows in the CSV.")
            # and seek back to the first row of the file (unless there's a header row)
            if not has_header:
                file.seek(0)
        database = []
        Item = namedtuple('Item', ['label', 'value'])
        # Ignore the probability label - I don't know what it's used for
        for row in csvreader:
            # Create a tuple of the label and the value of each item in the row
            # this will ensure different hashes in the case of values
            # from different columns coincidentally being equal
            transaction = {Item(labels[i], row[i]) for i in range(len(labels))}
            # Each transaction in the database is a set of such tuples
            database.append(transaction)
        return database


def join(l1, l2):
    """Join two itemsets that have all items equal except the last"""
    copy = list(l1)
    copy.append(l2[-1])
    return copy


def get_frequent_items(supports, min_sup):
    """Return a list of 1-itemsets that meet the minimum support"""
    return [[item] for item, count in supports.items() if count >= min_sup]


def get_supports(database):
    """Get all items in the database and their supports"""
    supports = {}
    for transaction in database:
        for item in transaction:
            # Use default of 0 for support.get() if item is not already counted
            supports[item] = supports.get(item, 0) + 1
    return supports


def has_infrequent_subset(candidate, itemsets, k):
    """Returns true if any (k-1)-subset of the candidate itemset is not an L_(k-1) frequent itemset"""
    return all(list(subset) not in itemsets for subset in combinations(candidate, k-1))


def apriori_candidates(itemsets, k):
    """Apriori method to generate candidate itemsets"""
    length = k - 2
    for l1 in itemsets:
        for l2 in itemsets:
            # If the itemsets are the same except for the last item
            if all(l1[i] == l2[i] for i in range(length)) and l1[length] < l2[length]:
                # Conveniently the candidate list will always be sorted
                # This is necessary for has_infrequent_subset() to work correctly
                candidate = join(l1, l2)
                if not has_infrequent_subset(candidate, itemsets, k):
                    yield candidate


def apriori(database, min_sup):
    """
    Run the Apriori algorithm on the database with the given minimum support

    @param: database A list of transactions, where each transaction is a list or set of hashable items.
    @param: min_sup The minimum support for an itemset to be considered frequent.

    @returns: itemsets_list A list of frequent k-itemsets that can indexed with itemsets_list[k]
    """
    itemsets_list = [[], get_frequent_items(get_supports(database), min_sup)]
    k = 2
    while itemsets_list[k-1]:
        itemsets_list.append([])
        # For each potential candidate
        for candidate in apriori_candidates(itemsets_list[k-1], k):
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


# TODO Refactor this a bit - I think insert_transaction() and insert_conditional_pattern()
# could be implemented a little bit more elegantly
class FPTree:
    def __init__(self, database, min_sup, supports=None):
        """Forms an FP-Tree given a dictionary of supports of frequent items and a list of transactions"""
        self.root = FPNode()
        self.header = {}
        self.min_sup = min_sup
        if supports is None:
            # We are calling from mine_patterns, rather than the main tree construction
            for conditional_pattern in database:
                self.insert_conditional_pattern(conditional_pattern)
        else:
            for transaction in database:
                self.insert_transaction(transaction, supports)

    def __str__(self):
        return self.root.subtree_to_string()[:-1]

    def __contains__(self, item):
        return item in self.header

    def insert_transaction(self, transaction, supports):
        """Given a transaction, trim it of items not meeting the minimum support and insert it into the tree"""
        trimmed_transaction = [item for item in transaction if supports[item] >= self.min_sup]
        # If the list is not empty, sort it and add it to the tree
        if trimmed_transaction:
            trimmed_transaction.sort(key=lambda item: supports[item], reverse=True)
            self.root.insert(trimmed_transaction, self.header)

    def insert_conditional_pattern(self, conditional_pattern):
        # Insert the conditional pattern into the tree, treating it like a transaction
        self.root.insert(conditional_pattern, self.header, conditional=True)

    def get_prefix_paths(self, item):
        """Given an item, get all paths from the root to nodes containing the item"""
        return [node.get_path_from_root() for node in self.header[item]]

    def mine_patterns(self, freq_itemsets=[], alpha=[]):
        """FP-Growth method of mining frequent patterns from the FP-Tree"""
        # TODO Should probably use sets for alpha and beta once everything else is solved
        for item, nodes in self.header.items():
            conditional_pattern_base = []
            support = sum(node.count for node in nodes)
            if support >= self.min_sup and item not in alpha:
                beta = [item] + alpha
                if beta not in freq_itemsets:
                    freq_itemsets.append(beta)
                conditional_pattern_base = self.get_prefix_paths(item)
                # TODO Trim out all nodes whose count do not meet the minimum supoprt!
                conditional_tree = FPTree(conditional_pattern_base, self.min_sup)
                conditional_tree.mine_patterns(freq_itemsets, beta)
        return freq_itemsets


class FPNode:
    def __init__(self, item=None, parent=None, count=1):
        self.children = []
        self.parent = parent
        self.item = item
        self.count = count

    def __contains__(self, item):
        return any(item == node.item for node in self.children)

    def is_root(self):
        return not self.parent

    def is_leaf(self):
        return not self.children

    def insert(self, items, header, conditional=False):
        """Given a sorted list of items that meet the minimum support, add the items down the tree.
        Also add the item, node pair to the tree's header for easy lookup of nodes corresponding to certain items."""
        if not items:
            return
        already_child_of_node = False
        if conditional:
            node = items[-1]
            item = node.item
            count = node.count
        else:
            item = items[-1]
            count = 1
        for child in self.children:
            if item == child.item:
                # Item is already a child of the current node, so increment its count
                # and insert the rest of the items into its subtree
                child.count += count
                child.insert(items[:-1], header, conditional)
                already_child_of_node = True
                break
        if not already_child_of_node:
            # Create a new subtree of the remaining items with the current node as the root
            child = FPNode(item, parent=self, count=count)
            if item in header:
                header[item].add(child)
            else:
                header[item] = {child}
            self.children.append(child)
            child.insert(items[:-1], header, conditional)

    def get_unique_leaf_node(self):
        """If there is only one child for every node from the root to the leaf, returns the unique leaf node
        otherwise returns None to indicate that there is more than one possible path down the tree"""
        if self.is_leaf():
            return self
        if len(self.children) > 1:
            return None
        # Children is a one-member set, so extract the only child
        (child,) = self.children
        return child.get_unique_leaf_node()

    def subtree_to_string(self, indent=0):
        """Returns a string representation of the subtree of the given node"""
        string = ""
        for i in range(indent):
            string += "\t"
        string += str(self.count) + ":" + str(self.item)
        string += "\n"
        for child in self.children:
            string += child.subtree_to_string(indent+1)
        return string


def fp_growth(database, min_sup):
    freq_itemsets = FPTree(database, min_sup, get_supports(database)).mine_patterns()
    # Sort and group itemsets by length and return
    freq_itemsets.sort(key=lambda l: len(l))
    return [[]] + [list(g) for k, g in groupby(freq_itemsets, key=len)] + [[]]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
