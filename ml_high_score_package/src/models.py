"""From-scratch regression models using NumPy only.

The project requirement forbids existing ML libraries.  These classes implement
learning algorithms directly and expose a small fit/predict interface.
"""
import numpy as np


class StandardScalerScratch:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RidgeLinearRegressionGD:
    """Linear regression with L2 regularization trained by batch gradient descent."""
    def __init__(self, learning_rate=0.03, epochs=3000, l2=1e-3, tolerance=1e-8, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.tolerance = tolerance
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        Xb = np.c_[np.ones(n), X]
        self.weights_ = np.zeros(p + 1)
        self.loss_history_ = []
        last_loss = np.inf
        for epoch in range(self.epochs):
            pred = Xb @ self.weights_
            error = pred - y
            grad = (Xb.T @ error) / n
            # Do not penalize the intercept.
            grad[1:] += self.l2 * self.weights_[1:]
            self.weights_ -= self.learning_rate * grad
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                loss = np.mean(error ** 2) / 2 + self.l2 * np.sum(self.weights_[1:] ** 2) / 2
                self.loss_history_.append(float(loss))
                if abs(last_loss - loss) < self.tolerance:
                    break
                last_loss = loss
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.weights_


class KNNRegressorScratch:
    """K-nearest-neighbor regression with Euclidean distance."""
    def __init__(self, k=15, batch_size=512):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float).reshape(-1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        k = min(self.k, len(self.X_train_))
        for start in range(0, len(X), self.batch_size):
            xb = X[start:start + self.batch_size]
            # Squared Euclidean distance, vectorized by batch.
            distances = np.sum((xb[:, None, :] - self.X_train_[None, :, :]) ** 2, axis=2)
            nn_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            preds.append(self.y_train_[nn_idx].mean(axis=1))
        return np.concatenate(preds)


class DecisionTreeRegressorScratch:
    """CART-style regression tree using variance/SSE reduction.

    To keep runtime reasonable for a course project, split candidates are quantiles
    rather than all unique values.
    """
    def __init__(self, max_depth=4, min_samples_leaf=30, n_thresholds=24, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_ = self.X_.shape[1]
        self.tree_ = self._build(self.X_, self.y_, depth=0)
        return self

    def _sse(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    def _best_split(self, X, y):
        n, p = X.shape
        if n < 2 * self.min_samples_leaf:
            return None, None, None
        base_sse = self._sse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        for j in range(p):
            col = X[:, j]
            if np.all(col == col[0]):
                continue
            qs = np.linspace(0.05, 0.95, self.n_thresholds)
            thresholds = np.unique(np.quantile(col, qs))
            for thr in thresholds:
                left_mask = col <= thr
                left_n = int(left_mask.sum())
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left_y = y[left_mask]
                right_y = y[~left_mask]
                gain = base_sse - self._sse(left_y) - self._sse(right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(thr)
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y)), "depth": depth, "n_samples": int(len(y))}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.var(y) < 1e-12:
            node["leaf"] = True
            return node
        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain is None or gain <= 1e-12:
            node["leaf"] = True
            return node
        mask = X[:, feature] <= threshold
        node.update({"leaf": False, "feature": int(feature), "threshold": float(threshold)})
        node["left"] = self._build(X[mask], y[mask], depth + 1)
        node["right"] = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, row, node):
        while not node.get("leaf", False):
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.tree_) for row in X], dtype=float)

    def _collect_importance(self, node, counts):
        if node.get("leaf", False):
            return
        counts[node["feature"]] += node.get("n_samples", 1)
        self._collect_importance(node["left"], counts)
        self._collect_importance(node["right"], counts)

    def feature_importances(self):
        counts = np.zeros(self.n_features_, dtype=float)
        self._collect_importance(self.tree_, counts)
        total = counts.sum()
        if total == 0:
            return counts
        return counts / total


class GradientBoostingRegressorScratch:
    """Gradient boosting for squared-error regression using small CART trees."""
    def __init__(self, n_estimators=40, learning_rate=0.08, max_depth=2, min_samples_leaf=40, n_thresholds=18):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.init_ = float(np.mean(y))
        pred = np.full_like(y, self.init_, dtype=float)
        self.trees_ = []
        self.training_loss_ = []
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_thresholds=self.n_thresholds,
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.trees_.append(tree)
            self.training_loss_.append(float(np.mean((y - pred) ** 2)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred

    def feature_importances(self):
        if not self.trees_:
            return None
        imps = np.array([tree.feature_importances() for tree in self.trees_])
        return imps.mean(axis=0)
