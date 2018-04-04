from __future__ import print_function
from argparse import ArgumentParser
from itertools import combinations, groupby
import csv


def init_parser():
    """Initialize the argument parser"""
    parser = ArgumentParser(description="Run Apriori Algorithm on a given transaction database.")
    parser.add_argument('filename',
                        help="Path to CSV file with transaction database.")
    parser.add_argument('--fp_growth', action='store_true',
                        help="Use the FP-Growth algorithm for frequent pattern mining rather than Apriori.")
    parser.add_argument('--min_sup', type=float, nargs='?', default=0.2,
                        help="Minimum support an itemset must meet to avoid being pruned. "
                             "Must be between 0 and 1. (default 0.2)")
    parser.add_argument('--labels', nargs='+',
                        help="List of labels for each column in the CSV. Only the first n columns of the CSV will be "
                             "read if n labels are specified and an error will be thrown if more labels are specified "
                             "than there are columns in the CSV. If no labels are specified, it is assumed that the "
                             "CSV contains a header with the labels of each column.")
    parser.add_argument('--count', action='store_true',
                        help="Just display the counts of each k-itemset, rather than printing the entire itemset.")
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    database = csv_to_transaction_db(args.filename, args.labels)

    if args.fp_growth:
        freq_itemsets = fp_growth(database, args.min_sup)
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
                labels = csvreader.next()
            else:
                # If it doesn't, print an error and exit
                exit("Error: it appears that the CSV file does not have a header row. "
                     "Please specify the labels of each column with --labels or add a header row to the CSV.")
        else:
            # Check if more labels were specified than there are rows in the CSV
            if len(labels) > len(file.readline()):
                exit("Error: more labels were specified than there are rows in the CSV.")
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


def get_freq_1_itemsets(supports, min_sup):
    """Return a list of 1-itemsets that meet the minimum support."""
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

    @returns: itemsets_list A list of frequent k-itemsets that can indexed with itemsets_list[k]
    """
    itemsets_list = [[], get_freq_1_itemsets(get_supports(database, min_sup))]
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


# Everything below here is poorly commented and ugly so far, sorry!!
class FPTree:
    def __init__(self, database, min_sup, supports=None, beta=None):
        """Forms an FP-Tree given a dictionary of supports of frequent items and a list of transactions"""
        self.root = FPNode()
        self.header = {}
        self.min_sup = min_sup
        if supports is None:
            for transaction in database:
                self.root.insert(transaction, self.header)
        else:
            for transaction in database:
                self.insert(transaction, supports)

    def __str__(self):
        return self.root.subtree_to_string()[:-1]

    def __contains__(self, item):
        return item in self.header

    def insert(self, transaction, supports):
        """Given a transaction, trim it of items not meeting the minimum support and insert it into the tree"""
        trimmed_transaction = [item for item in transaction if supports[item] >= self.min_sup]
        # If the list is not empty, sort it and add it to the tree
        if trimmed_transaction:
            trimmed_transaction.sort(key=lambda item: supports[item], reverse=True)
            self.root.insert(trimmed_transaction, self.header)

    def get_unique_leaf_node(self):
        """Gets the unique leaf node if one exists - i.e. the tree only has one path, otherwise returns None."""
        return self.root.get_unique_leaf_node()

    def mine_patterns(self, alpha=set()):
        """FP-Growth method of mining frequent patterns from the FP-Tree"""
        leaf = self.get_unique_leaf_node()
        freq_itemsets = []
        if leaf:
            single_prefix_path = leaf.get_prefix_path_from_root()
            for i in range(1, len(single_prefix_path)+1):
                for subset in (combinations(single_prefix_path, i)):
                    # subset = set(subset) | alpha
                    support = min(node.count for node in subset)
                    if support >= self.min_sup:
                        freq_itemsets.append(list(subset))
        else:
            for item, nodes in self.header.items():
                conditional_pattern_base = []
                support = sum(node.count for node in nodes)

                if support >= self.min_sup:
                    for node in nodes:
                        pattern = []

                        for n in node.get_prefix_path_from_root():
                            if n.count >= self.min_sup:
                                pattern.append(n.item)

                        conditional_pattern_base.append(pattern)

                    beta = set(alpha)
                    beta.add(item)

                    conditional_tree = FPTree(conditional_pattern_base, self.min_sup, beta=beta)

                    freq_itemsets.extend(conditional_tree.mine_patterns(beta))
        return freq_itemsets


class FPNode:
    def __init__(self, item=None, parent=None):
        self.children = []
        self.parent = parent
        self.item = item
        self.count = 1

    def is_root(self):
        return not self.parent

    def is_leaf(self):
        return not self.children

    def insert(self, sorted_items, header):
        """Given a sorted list of transactions that meet the minimum support, add the items down the tree.
        Also add the item, node pair to the tree's header for easy lookup of nodes corresponding to certain items."""
        if not sorted_items:
            return
        already_child_of_node = False
        item = sorted_items[-1]
        for child in self.children:
            if item == child.item:
                # Item is already a child of the current node, so increment its count
                # and insert the rest of the items into its subtree
                child.count += 1
                child.insert(sorted_items[:-1], header)
                already_child_of_node = True
                break
        if not already_child_of_node:
            # Create a new subtree of the remaining items with the current node as the root
            child = FPNode(item, parent=self)
            if item in header:
                header[item].add(child)
            else:
                header[item] = {child}
            self.children.append(child)
            child.insert(sorted_items[:-1], header)

    def get_unique_leaf_node(self):
        """If there is only one child for every node from the root to the leaf, returns the unique leaf node
        otherwise returns None to indicate that there is more than one possible path down the tree"""
        if self.is_leaf():
            return self
        if len(self.children) > 1:
            return None
        child = tuple(self.children)[0]
        return child.get_unique_leaf_node()

    def get_prefix_path_from_root(self):
        """Starting from a node, builds a list of nodes between the node and the root of the tree"""
        prefix_path = []
        node = self
        while not node.is_root():
            prefix_path.append(node.parent)
            node = node.parent
        prefix_path.reverse()
        return prefix_path


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
    min_sup *= len(database)
    supports = get_supports(database)

    tree = FPTree(database, min_sup, supports)

    print(tree)

    # print()
    # entry = tree.header[('capital-loss', '0')]
    # for node in entry[1]:
    #     if node.count == 87:
    #         break
    # for node in node.get_prefix_path_from_root():
    #     print(node.count, ":", node.item)

    # exit(0)

    freq_itemsets = tree.mine_patterns()
    print(freq_itemsets)

    # Sort and group itemsets by length and return
    freq_itemsets.sort(key=lambda l: len(l))
    return [[]] + [list(g) for k, g in groupby(freq_itemsets, key=len)]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
