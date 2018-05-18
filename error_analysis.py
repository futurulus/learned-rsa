#!/usr/bin/env python

import sys
import cPickle as pickle
from collections import defaultdict, Counter
from operator import itemgetter
import numpy as np
from metrics import instance_accuracy
from utils import confidence_interval

def predicted_proper_subset(prediction, actual):
    """Multiset proper subset"""
    p = Counter(prediction)
    a = Counter(actual)
    for x, c in p.items():
        if c > a[x]:
            return False
    for x, c in a.items():
        if c > p[x]:
            return True
    return False



    

def predicted_proper_superset(prediction, actual):
    """Multiset proper superset"""
    p = Counter(prediction)
    a = Counter(actual)
    for x, c in p.items():
        if c < a[x]:
            return False
    for x, c in a.items():
        if c < p[x]:
            return True
    return False
    

def predicted_proper_subset_stats(log, agentname='pragmatic', overproduce=False):
    """Counts how many predictions are properly contained in the actual value"""
    counts = defaultdict(int)
    relation = predicted_proper_superset if overproduce else predicted_proper_subset
    for d in log:
        results = d[agentname]        
        outcome = relation(results['prediction'], results['actual'])
        outcome = 'subset' if outcome else 'not subset'
        evaluation = instance_accuracy(results['prediction'], results['actual'])
        evaluation = 'correct' if evaluation else 'incorrect'
        counts[(evaluation, outcome)] += 1
    target_value = float(counts[('incorrect', 'subset')])
    total = float(sum([counts[('incorrect', x)] for x in ('subset', 'not subset')]))
    acc = (target_value / total)*100
    # print 'proper subset: %d of %d incorrect predictions (%0.02f%%)' % (target_value, total, acc)
    return (target_value, total, acc)


def what_get_left_out(prediction, actual):
    return set(actual) - set(prediction)


def what_get_left_out_stats(log, agentname='pragmatic'):
    """Counts how many predictions are properly contained in the actual value"""
    counts = defaultdict(int)
    for d in log:
        results = d[agentname]        
        outcome = what_get_left_out(results['prediction'], results['actual'])
        for x in outcome:
            counts[x] += 1
    for key, val in sorted(counts.items(), key=itemgetter(1), reverse=True)[:5]:
        print key, val


def predicted_vs_actual_length(log, agentname='pragmatic'):
    deltas = []
    for d in log:
        results = d[agentname]
        delta = len(results['prediction']) - len(results['actual'])
        deltas.append(delta)
    upper, lower = confidence_interval(deltas)
    print '%s mean difference: %0.02f (%0.02f, %0.02f 95%% ci)' % (agentname, np.mean(deltas), upper, lower)


if __name__ == '__main__':

    # RSA proper subset counts report:
    # for domain in ('furniture', 'people'):
    #     print domain
    #     predicted_proper_subset_stats(pickle.load(file('logs/log_%s.pickle' % domain)))

    # for domain in ('furniture', 'people'):
    #     print domain
    #     for agentname in ('literal', 'pragmatic'):
    #         predicted_vs_actual_length(pickle.load(file('logs/log_%s.pickle' % domain)), agentname=agentname)

    for domain in ('furniture', 'people'):
        print "======================================================================"
        print domain
        what_get_left_out_stats(pickle.load(file('logs/log_%s.pickle' % domain)))
