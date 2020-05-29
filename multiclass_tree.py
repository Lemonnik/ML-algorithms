import numpy as np
import sys
import math
import queue

class TreeNode:
    def __init__(self, feat = ''):
        self.feature = feat
        self.values = None
        self.branches = None


class DecisionTree:
    def __init__(self, data=None):
        if data:
            # self.train, self.test = self.separate_data(np.matrix(data))
            self.train = np.array(data)
            self.classes = np.unique(self.train[1:, -1])
            # features and their values
            self.features = []
            for col in self.train.T[:-1]:
                self.features.append([col[0]] + np.unique(col[1:]).tolist())
            # remove 1st row with feature names
            self.train = np.delete(self.train, (0), axis=0)
            root = self.build_tree(self.train, self.features)
            l = self.levels(root)
            self.print_tree(l)


    def separate_data(self, data):
        total = data.shape[0]
        ct1 = int(2 / 3 * total)  # train size

        ind = np.random.permutation(total) 
        train_ind = ind[:ct1]  # train sample
        test_ind = ind[ct1:total]  # test sample

        train = data[train_ind, :]
        test = data[test_ind, :]

        return train, test

    # id3 with IGain split criterion
    def id3(self, data, features):
        # sample entropy
        h_sample = 0.0
        total = data.shape[0]
        for q in self.classes:
            # number of elements of each class
            quantity = np.count_nonzero(data[:, -1] == q)
            if quantity:
                h_sample -= quantity / total * np.log2(quantity / total)
        # pairs {feature numbers, gain value}
        IGain = [0, 0]
        for feature in features:
            col = features.index(feature)
            h = 0.0  # entropy for current feature
            for x in feature[1:]:
                base = len(feature[1:])
                h_feat = 0.0
                ind = np.argwhere(data[:, col] == x)[:, 0]
                count = len(ind)
                if count:
                    for y in self.classes:
                        class_count = np.count_nonzero(data[ind, -1] == y)
                        if class_count:
                            h_feat -= class_count / count * np.log2(class_count / count)
                h += count / total * h_feat
            gain = h_sample - h
            if IGain[1] < gain:
                IGain[0] = col  # feature number
                IGain[1] = gain  # IGain value
        return IGain[0]

    def build_tree(self, data, features):
        t = TreeNode()
        if len(features):
            # feature selection
            num = self.id3(data, features)
            t.feature = features[num][0]
            t.values = [None] * len(features[num][1:])
            t.branches = [TreeNode()] * len(features[num][1:])
            i = 0
            for x in features[num][1:]:
                ind = np.argwhere(data[:, num] == x)[:, 0]
                t.values[i] = x
                if not ind.size:
                    continue
                # all objects have the same class
                if np.unique(data[ind, -1]).shape[0] == 1:
                    t.branches[i] = TreeNode(data[ind[0], -1])
                else:
                    d = data[ind, :]
                    d = np.delete(d, (num), axis=1)
                    t.branches[i] = self.build_tree(d, features[:num] + features[num+1:])
                i += 1
        return t

    # tree levels for print
    def levels(self, root):
        m = []
        cur_level = queue.Queue()
        next_level = queue.Queue()
        cur_level.put(root)
        # modified BFS
        while True:
            m.append([])
            for i in range(cur_level.qsize()):
                cur = cur_level.get()
                if cur.branches:
                    for branch in cur.branches:
                        next_level.put(branch)
                else:
                    next_level.put(TreeNode('null'))
                if cur.values:
                    s = cur.feature + ' (' + '; '.join(cur.values) + ')'
                    m[-1].append(s)
                else:
                    m[-1].append(cur.feature)
            if all(v == 'null' for v in m[-1]):
                del m[-1]
                break
            cur_level = next_level
        return m

    def print_tree(self, m):
        space = '  '
        i = len(m)
        print(space * 3 * i, end=space * i)
        for level in m:
            print(space * i, end=space * i)
            for node in level:
                if node != 'null':
                    print('{:<10}'.format(node), end = space * i)
                else:
                    print('{:<10}'.format(" "), end = space * i)
            print('\n')
            i -= 1


data = []
for line in sys.stdin:
    data.append(line.split())

tree = DecisionTree(data)
