from node import Node
import math
import random
import numpy as np
from collections import defaultdict, OrderedDict
import pdb

class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_impurity_split=1e-07, splitter='best', max_features=None):
        
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.splitter = splitter
        self.root = None

    def entropy(hist):
        total = sum(hist.values())
        return -sum([c/total * math.log(c/total, 2) if c > 0 else 0 for c in hist.values()])
    
    def impurity(left_label_hist, right_label_hist):
        left_entropy = DecisionTree.entropy(left_label_hist)
        right_entropy = DecisionTree.entropy(right_label_hist)
        left_count = sum(left_label_hist.values())
        right_count = sum(right_label_hist.values())
        total = left_count + right_count
        return (left_count * left_entropy + right_count * right_entropy)/total

    # Returns the split rule
    def segmenter(self, data, labels):
        min_impurity = math.inf
        best_feature_index = None
        best_feature_thresh = None
        if self.max_features is not None:
            lim = random.sample(range(data.shape[1]), k=self.max_features)
        else:
            lim = range(data.shape[1])
        for feature_index in lim:
            splits = defaultdict(lambda: defaultdict(int))
            right_hist = defaultdict(int)
            for i in range(data.shape[0]):
                splits[data[i][feature_index]][labels[i]] += 1
                right_hist[labels[i]] += 1
            left_hist = defaultdict(int)
            split_min = math.inf
            best_thresh = None
            splits = OrderedDict(sorted(splits.items(), key=lambda t: t[0]))
            for threshold, counts in splits.items():
                for label, count in counts.items():
                    left_hist[label] += count
                    right_hist[label] -= count
                dirtiness = DecisionTree.impurity(left_hist, right_hist)
                if dirtiness < split_min:
                    split_min = dirtiness
                    best_thresh = threshold
            if split_min < min_impurity:
                min_impurity = split_min
                best_feature_thresh = best_thresh
                best_feature_index = feature_index
        x, counts = np.unique(labels, return_counts=True)
        hs = -sum([c/len(labels) * math.log(c/len(labels), 2) for c in counts])
        if hs - min_impurity < self.min_impurity_split:
            return None
        if best_feature_thresh is not None and min_impurity > self.min_impurity_split:
            return (best_feature_index, best_feature_thresh)
        else:
            return None

    # Checks for max_depth, min_samples_split, and homogeneity. Returns the majority vote or None.
    def terminate(self, labels, depth):
        classes, counts = np.unique(labels, return_counts=True)
        if (self.max_depth is not None and depth >= self.max_depth) \
            or len(labels) < self.min_samples_split or len(classes) < 2:
            return classes[np.argmax(counts)]
        return None

    def split(self, arr, cond):
        return arr[cond], arr[~cond]

    def grow_tree(self, data, labels, depth):
        leaf_cond = self.terminate(labels, depth)
        if leaf_cond is not None:
            return Node(depth, data.shape[0], label=leaf_cond)
        rule = self.segmenter(data, labels)
        if rule is None:
            x, counts = np.unique(labels, return_counts=True)
            return Node(depth, data.shape[0], label=labels[np.argmax(counts)])
        combined = np.append(data, labels[np.newaxis].T, axis=1)
        left_data, right_data = self.split(combined, combined[:, rule[0]] <= rule[1])
        left_labels, left_data = left_data[:, -1], np.delete(left_data, -1, axis=1)
        right_labels, right_data = right_data[:, -1], np.delete(right_data, -1, axis=1)
        left_child = self.grow_tree(left_data, left_labels, depth+1)
        right_child = self.grow_tree(right_data, right_labels, depth+1)
        return Node(
                    depth, data.shape[0], split_rule=rule,
                    left_child=left_child,
                    right_child=right_child,
                    )

    def train(self, data, labels):
        if self.splitter == 'random' and self.max_features is None or \
            self.max_features is not None and self.max_features > data.shape[1]:
            self.max_features = int(math.sqrt(data.shape[1]))
        self.root = self.grow_tree(data, labels, 0)

    def predict(self, data):
        if self.root is None:
            raise AttributeError('Tree has not been built.')
        curr_node = self.root
        while curr_node.label is None:
            curr_node = curr_node.traverse(data)
        return curr_node.label

    def visual_node(self, node):
        if node.label is None:
            return 'depth: {}\npoints: {}\nlabel:{}'.format(node.depth, node.points, node.label)
        else:
            return 'depth: {}\npoints: {}\nthreshold: {}\nindex: {}'.format(node.depth, node.points, node.split_rule[1], node.split_rule[0])

    # def visualize(self, graph=None, parent=None, node=None):
    #     if graph is None:
    #         graph = AGraph(directed=True)
    #         n = visual_node(self.root)
    #         graph.add_node(n)
    #     else:
    #         n = visual_node(node)
    #         graph.add_node(n)
    #         graph.add_edge(parent, n)
    #     if n.label is None:
    #         visualize(graph, )
