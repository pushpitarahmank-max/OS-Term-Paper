"""Lightweight Machine Learning Models for Page Eviction Prediction.

Implemented using pure NumPy to guarantee zero external C-dependencies
while delivering high numerical stability and interpretable weights.
"""

from typing import Tuple, Optional
import numpy as np


class LogisticRegressionClassifier:
    """Lightweight Logistic Regression with L2 Regularization.

    Predicts whether a page in cache is an ideal eviction candidate
    (i.e., has a large forward reuse distance).
    """

    def __init__(self, learning_rate: float = 0.05, n_epochs: int = 50, l2_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Fit weights using gradient descent with standardized features."""
        n_samples, n_features = X.shape
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-7
        X_scaled = (X - self.mean) / self.std

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            linear_pred = np.dot(X_scaled, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            dw = (1.0 / n_samples) * np.dot(X_scaled.T, (predictions - y)) + self.l2_reg * self.weights
            db = (1.0 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of being an eviction candidate."""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        X_scaled = (X - self.mean) / self.std
        linear_pred = np.dot(X_scaled, self.weights) + self.bias
        probs = self._sigmoid(linear_pred)
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class SimpleDecisionTreeNode:
    """Node for a lightweight Decision Tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifierCustom:
    """Lightweight Decision Tree Classifier for Eviction Prediction.

    Uses Gini impurity with constrained max_depth to avoid overfitting
    and ensure deterministic, sub-microsecond inference.
    """

    def __init__(self, max_depth: int = 4, min_samples_split: int = 4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[SimpleDecisionTreeNode] = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p1 = np.sum(y == 1) / len(y)
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_gini = self._gini(y)

        n_samples, n_features = X.shape
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            if len(thresholds) > 20:
                # Subsample quantiles for efficiency
                thresholds = np.percentile(X_column, np.linspace(5, 95, 15))

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                gini_l = self._gini(y[left_mask])
                gini_r = self._gini(y[right_mask])
                child_gini = (n_l / n_samples) * gini_l + (n_r / n_samples) * gini_r

                gain = current_gini - child_gini
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> SimpleDecisionTreeNode:
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Leaf criteria
        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            leaf_value = float(np.mean(y)) if len(y) > 0 else 0.5
            return SimpleDecisionTreeNode(value=leaf_value)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            leaf_value = float(np.mean(y)) if len(y) > 0 else 0.5
            return SimpleDecisionTreeNode(value=leaf_value)

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return SimpleDecisionTreeNode(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierCustom":
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _traverse(self, x: np.ndarray, node: SimpleDecisionTreeNode) -> float:
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        return np.array([self._traverse(x, self.root) for x in X])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
