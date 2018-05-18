import numpy as np
from collections import Counter

def multiset_dice(a, b):
    d1 = Counter(a)
    d2 = Counter(b)
    num = 2.0 * sum([min([d1[x], d2[x]]) for x in d1])
    den = float(len(a) + len(b))
    return num / den

def instance_accuracy(a, b):
    return 1.0 if a == b else 0.0


def mean_multiset_dice(x, y):
    return np.mean([multiset_dice(a,b) for a, b in zip(x,y)])


def accuracy(x, y):
    return sum([1.0 for a, b in zip(x,y) if instance_accuracy(a, b)]) / len(x)


def max_multiset_dice(x, y):
    """Should be the same as accuracy"""
    return sum([1.0 for a, b in zip(x,y) if multiset_dice(a,b)==1.0]) / len(x)


