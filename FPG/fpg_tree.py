from queue import Queue
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from copy import deepcopy

class Sorting:
    def __init__(self, file=None):
        # support levels for each item
        self.freq = {}
        # items sorted alphabetically and support level
        # (list of tuples)
        self.sorted_freq = []
        # transaction list
        self.transactions = []
        if file:
            self.get_list(file)
        self.compute_support()
        self.sort_freq()
        self.sort_transactions()

    def get_list(self, file):
        f = open(file, 'r')
        for line in f.readlines():
            self.transactions.append(line.split())
        f.close()

    def compute_support(self):
        for t in self.transactions:
            for word in t:
                if word not in self.freq:
                    self.freq[word] = 1
                else:
                    self.freq[word] += 1


    def sort_freq(self):
        sorted_freq = sorted(self.freq.items(), key=lambda x: x[0], reverse=True)
        self.sorted_freq = sorted(sorted_freq, key=lambda x: x[1])

    def sort_transactions(self):
        for t in self.transactions:
            i = len(t) - 1
            for elem in self.sorted_freq:
                if elem[0] in t:
                    ind = t.index(elem[0])
                    tmp = t[ind]
                    t[ind] = t[i]
                    t[i] = tmp
                    i -= 1


# tree node
class Node:
    def __init__(self, name):
        self.name = name 
        self.freq = 1  # support level
        self.num = 0  # item number (for transformation into graph)
        self.leaves = []  # потомки


class Tree:
    def __init__(self, sorted_transactions):
        self.root = Node('R')
        for t in sorted_transactions:
            self.build_tree(self.root, t)
        self.enumerate_nodes()
        self.graph = nx.DiGraph()
        self.graph.add_node(self.root.num, data=self.root.name, freq=self.root.freq)
        self.build_graph(self.root)
        # self.draw_graph()


    def build_branch(self, transaction):
        node = Node(transaction[0])
        if len(transaction) > 1:
            node.leaves.append(self.build_branch(transaction[1:]))
        return node


    def find_match(self, node, name):
        for i in range(len(node.leaves)):
            if node.leaves[i].name == name:
                node.leaves[i].freq += 1
                return i
        return -1


    def build_tree(self, root, transaction):
        ind = self.find_match(root, transaction[0])
        if ind != -1 and len(transaction) > 1:
            self.build_tree(root.leaves[ind], transaction[1:])
        elif ind == -1:
            root.leaves.append(self.build_branch(transaction))

    # DFS
    def enumerate_nodes(self):
        q = Queue()
        q.put(self.root)
        ct = 0
        while q.qsize():
            elem = q.get()
            elem.num = ct
            ct += 1
            for leaf in elem.leaves:
                q.put(leaf)


    def build_graph(self, root):
        for leaf in root.leaves:
            self.graph.add_node(leaf.num, data=leaf.name, freq=leaf.freq)
            self.graph.add_edge(root.num, leaf.num)
            self.build_graph(leaf)


    def draw_graph(self):
        write_dot(self.graph, 'test.dot')
        pos = graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos, with_labels=False, arrows=True, node_color='r', alpha=0.3, node_size=700)
        freq = nx.get_node_attributes(self.graph, 'freq')
        nx.draw_networkx_labels(self.graph, pos, labels=freq, font_weight='bold')
        names = nx.get_node_attributes(self.graph, 'data')
        nx.draw_networkx_labels(self.graph, pos, labels=names, font_size=6)
        #plt.show()


class ConditionalTree:
    def __init__(self, tree, name, support_level=1):
        self.root = Node('R')
        self.name = name
        self.basis = tree
        self.paths = []
        self.freq = {}
        self.sorted_freq = []
        self.graph = nx.DiGraph()
        self.level = support_level

        self.search_paths(self.basis.root, [], self.name)
        print('Paths to node \'', name, "\':")
        print(self.paths)
        self.remove_suffix()
        if self.paths:
            self.compute_support()
            self.sort_freq()
            self.sort_transactions()
            self.transactions = deepcopy(self.paths)
            # conditional basis
            for p in self.transactions:
                self.build_tree(self.root, p)
            self.enumerate_nodes()
            self.graph.add_node(self.root.num, data=self.root.name, freq=self.root.freq)
            self.build_graph(self.root)
            # self.draw_graph()
            res = self.check_support()
            print('Popular items: ')
            print(res)
        else:
            print('After removing suffix to \'', self.name, '\' no paths left.')


    def search_paths(self, root, path, name):
        if root:
            if root.name != name:
                for leaf in root.leaves:
                    self.search_paths(leaf, path + [root.name], name)
            else:
                self.paths.append(path[1:] + [root.name, root.freq])


    def remove_suffix(self):
        for i in range(len(self.paths)):
            if len(self.paths[i]) > 2:
                self.paths[i].remove(self.name)
            else:
                del self.paths[i]


    def compute_support(self):
        for p in self.paths:
            for name in p[:-1]:
                if name in self.freq:
                    self.freq[name] += p[-1]
                else:
                    self.freq[name] = p[-1]

    def sort_freq(self):
        sorted_freq = sorted(self.freq.items(), key=lambda x: x[0], reverse=True)
        self.sorted_freq = sorted(sorted_freq, key=lambda x: x[1])


    def sort_transactions(self):
        for p in self.paths:
            i = len(p) - 2
            for elem in self.sorted_freq:
                if elem[0] in p:
                    ind = p.index(elem[0])
                    tmp = p[ind]
                    p[ind] = p[i]
                    p[i] = tmp
                    i -= 1

    def build_branch(self, transaction):
        node = Node(transaction[0])
        node.freq = transaction[-1]
        if len(transaction) > 2:
            node.leaves.append(self.build_branch(transaction[1:]))
        return node

    @staticmethod
    def find_match(node, name, ct=1):
        for i in range(len(node.leaves)):
            if node.leaves[i].name == name:
                node.leaves[i].freq += ct
                return i
        return -1

    def build_tree(self, root, transaction):
        ind = self.find_match(root, transaction[0], transaction[-1])
        if ind != -1 and len(transaction) > 2:
            self.build_tree(root.leaves[ind], transaction[1:])
        elif ind == -1:
            root.leaves.append(self.build_branch(transaction))

    def enumerate_nodes(self):
        q = Queue()
        q.put(self.root)
        ct = 0
        while q.qsize():
            elem = q.get()
            elem.num = ct
            ct += 1
            for leaf in elem.leaves:
                q.put(leaf)

    def build_graph(self, root):
        for leaf in root.leaves:
            self.graph.add_node(leaf.num, data=leaf.name, freq=leaf.freq)
            self.graph.add_edge(root.num, leaf.num)
            self.build_graph(leaf)


    def draw_graph(self):
        write_dot(self.graph, 'test.dot')
        pos = graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos, with_labels=False, arrows=True, node_color='y', alpha=0.3, node_size=70)
        freq = nx.get_node_attributes(self.graph, 'freq')
        nx.draw_networkx_labels(self.graph, pos, labels=freq, font_weight='bold')
        names = nx.get_node_attributes(self.graph, 'data')
        nx.draw_networkx_labels(self.graph, pos, labels=names, font_size=6)
        #plt.show()


    def check_support(self):
        res = []
        visited = []
        # BFS
        q = Queue()
        q.put(self.root)
        while q.qsize():
            elem = q.get()
            # non-root node
            if elem.name != 'R' and elem.name not in visited:
                visited.append(elem.name)
                self.paths.clear()
                self.search_paths(self.root, [], elem.name)
                self.paths=sorted(self.paths, key=lambda x: len(x), reverse=True)
                for i in range(len(self.paths)):
                    if i >0 :
                        self.paths[i][-1] += self.paths[i-1][-1]
                    if self.paths[i][-1] >= self.level:
                        self.paths[i].insert(-1, self.name)
                        res.append(self.paths[i])
            for leaf in elem.leaves:
                q.put(leaf)
        return res


s = Sorting(file='data')
t = Tree(s.transactions)

plt.figure(1)
plt.title('Basis')
plt.subplot(111)
t.draw_graph()
cond_trees = []
for name in s.freq:
    cond_trees.append(ConditionalTree(t, name, 1))

i=4;j=3;k=1
plt.figure(2)
plt.title('Conditional basis')
for tree in cond_trees:
    if tree.graph:
        ax = plt.subplot(i, j, k)
        ax.set_title(tree.name, fontsize=6, fontweight='bold')
        k += 1
        tree.draw_graph()
plt.show()
