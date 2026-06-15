import numpy as np
from collections import Counter


class Node:
    # First we initialize the parameters required
    def __init__(self, features=None, threshold=None, left=None, right=None, *,value=None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Checking if the node is leaf node or not.
    def is_leaf_node(self):
        return self.value is not None 

class DecisionTree:
    # Initializing the parameters required for the decision tree
    def __init__(self, min_samples_split=10, max_depth=100, n_features=None, root=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = root

    # fit function which will take values of X_train and y_train
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Checking the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Finding the best split
        best_features, best_thres = self._best_split(X, y, feat_idxs)

        if best_features is None:
            return Node(value=self.most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_features], best_thres)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth= depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth= depth+1)
        return Node(best_features, best_thres, left, right)
    

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx = None
        split_threshold= None


        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

        
            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
        
    def _information_gain(self, y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs)== 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r 
        
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, split_thres):
        left_idxs = np.argwhere(X_column <= split_thres).flatten()
        right_idxs = np.argwhere(X_column > split_thres).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])


    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.features]<= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)