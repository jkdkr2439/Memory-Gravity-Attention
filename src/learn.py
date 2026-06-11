"""
Learning theta for MGA retrievers (§5.1)
Logistic regression on training queries.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple


def learn_theta_gate(train_features: List[np.ndarray], 
                     train_labels: List[np.ndarray]) -> np.ndarray:
    """
    Learn theta for mga_gate: sim × (1 + sigmoid(theta^T f_persistent))
    
    Training: for each query, gold nodes = positive, non-gold = negative.
    Features: persistent features only (cols 1-6, excluding similarity).
    
    Returns theta vector (6,)
    """
    X_all = []
    y_all = []
    
    for features, labels in zip(train_features, train_labels):
        f_persistent = features[:, 1:]  # exclude similarity column
        X_all.append(f_persistent)
        y_all.append(labels)
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    if len(np.unique(y)) < 2:
        # Not enough signal, return uniform
        return np.ones(X.shape[1]) * 0.5
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    theta = clf.coef_[0]
    
    return theta


def learn_theta_linear(train_features: List[np.ndarray],
                       train_labels: List[np.ndarray]) -> np.ndarray:
    """
    Learn theta for mga_linear: theta^T features (all 7 features).
    Returns theta vector (7,)
    """
    X_all = []
    y_all = []
    
    for features, labels in zip(train_features, train_labels):
        X_all.append(features)
        y_all.append(labels)
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    if len(np.unique(y)) < 2:
        return np.ones(X.shape[1]) * 0.5
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    theta = clf.coef_[0]
    
    return theta
