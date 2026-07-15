"""Compat loader for logisticclassifier.pkl, pickled under Python 2 / pre-0.18 scikit-learn.

sklearn.linear_model.logistic was renamed to sklearn.linear_model._logistic, so the
pickle's class reference needs a module alias to resolve. The pickle also contains
Python 2 str byte strings, which need latin1 decoding under Python 3.
"""

import pickle
import sys

from sklearn.linear_model import _logistic

sys.modules.setdefault("sklearn.linear_model.logistic", _logistic)


def load_logistic_classifier(path="logisticclassifier.pkl"):
    with open(path, "rb") as fid:
        return pickle.load(fid, encoding="latin1")
