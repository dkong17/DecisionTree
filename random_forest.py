import numpy as np
from node import Node
from sklearn.utils import resample
from decision_tree import DecisionTree
from collections import defaultdict
import operator

class RandomForest:

    def __init__(self, n_estimators=10, max_depth=None, max_features=None):
        self.trees = []
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth

    def train(self, data, labels):
        for i in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, splitter='random', max_features=self.max_features)
            sample_data, sample_labels = resample(data, labels)
            tree.train(sample_data, sample_labels)
            self.trees.append(tree)

    def predict(self, data):
        predictions = defaultdict(int)
        for tree in self.trees:
            label = tree.predict(data)
            predictions[label] += 1
        return max(predictions.items(), key=operator.itemgetter(1))[0]