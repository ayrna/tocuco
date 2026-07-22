# -*- coding: utf-8 -*-
"""
Scorers for cross-validation.

Each entry in SCORERS maps a name to a dict with:
  - fn              : callable(y_true, y_pred) -> float
  - greater_is_better: bool (passed to make_scorer)

Usage
-----
from experiments.utils.scorers import get_scorer

scorer = get_scorer("bacc")   # or "amae"
GridSearchCV(estimator, ..., scoring=scorer)
"""

from dlordinal.metrics import amae
from sklearn.metrics import balanced_accuracy_score, make_scorer

SCORERS = {
    "amae": {"fn": amae, "greater_is_better": False},
    "bacc": {"fn": balanced_accuracy_score, "greater_is_better": True},
}


def get_scorer(name: str):
    """Return a sklearn scorer for the given metric name.

    Parameters
    ----------
    name : str
        One of the keys in SCORERS (e.g. ``"amae"``, ``"bacc"``).

    Returns
    -------
    scorer : sklearn scorer object
    """
    name = name.lower()
    if name not in SCORERS:
        raise ValueError(f"Unknown scorer '{name}'. Available: {list(SCORERS)}")
    entry = SCORERS[name]
    return make_scorer(entry["fn"], greater_is_better=entry["greater_is_better"])
