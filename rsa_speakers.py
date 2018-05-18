#!/usr/bin/env python

import sys
from glob import glob
import cPickle as pickle
import itertools
from collections import defaultdict, Counter
from operator import itemgetter
import random
import numpy as np
import json

import training_instances
from metrics import accuracy, multiset_dice, instance_accuracy
from utils import rownorm, safelog, confidence_interval

from sklearn.cross_validation import KFold, ShuffleSplit

def all_zero_costs(msgs):
    return np.repeat(0.0, len(msgs))

def linear_in_length(msgs):
    return np.array([float(len(msg)-1) for msg in msgs])

def inverse_linear_in_length(msgs):
    return np.array([-float(len(msg)-1) for msg in msgs])
    

class Experiment:
    def __init__(self, filenames=glob('../TUNA/corpus/singular/furniture/*.xml'), cv=5, test_size=None, logfile=None):
        self.filenames = filenames
        self.cv = cv
        self.test_size = test_size
        # Hyper-parameters optimized on training data:
        self.temp_range = np.arange(0.1, 1.1, 0.1)
        self.nullcost_range = np.arange(0.0, 2.0, 1.0)
        self.cost_functions = [all_zero_costs, linear_in_length, inverse_linear_in_length]
        # Logging:
        if logfile:
            self.logfile = logfile
            self.log = []
        
    def run(self):
        D = training_instances.get_generation_instances(filenames=self.filenames)
        splits = None
        if self.test_size:
            splits = ShuffleSplit(n=len(D), n_iter=self.cv, test_size=self.test_size)
        else:
            splits = KFold(n=len(D), n_folds=self.cv, shuffle=True)
        cross_val_results = defaultdict(list)
        for fold_index, (train_indices, test_indices) in enumerate(splits):
            train = [D[i] for i in train_indices]
            test = [D[i] for i in test_indices]
            params = self.set_hyperparameters(train)            
            run_results = self.crossval_run(test, params, fold_index=fold_index)
            print "======================================================================"
            print params
            print run_results
            for key, val in run_results.items():
                cross_val_results[key].append(val)
        for key, vals in sorted(cross_val_results.items()):
            modelname, metricname = key
            lower, upper = confidence_interval(vals)
            print "%s mean %s: %0.03f (%0.03f-%0.03f)" % (modelname, metricname, np.mean(vals), lower, upper)
        pickle.dump(self.log, file(self.logfile, 'w'), 2)
            
    def crossval_run(self, data, params, logging=True, fold_index=None):
        run_results = defaultdict(list)
        for x in data:
            mod = SpeakerRSA(x, **params)
            trial_results = mod.evaluate()            
            if logging:
                trial_results['fold_index'] = fold_index
                self.log.append(trial_results)
            for agentname, vals in trial_results.items():
                if agentname != 'fold_index':
                    for metricname, val in vals['evaluations'].items():
                        run_results[(agentname, metricname)].append(val)
        # Return the means:
        return {key: np.mean(val) for key, val in run_results.items()}
                
    def set_hyperparameters(self, data):
        results = []        
        for temp, nullcost, cost_function in itertools.product(self.temp_range, self.nullcost_range, self.cost_functions):
            params = {'temperature':temp, 'nullcost':nullcost, 'cost_function':cost_function}
            all_reports = self.crossval_run(data, params, logging=False)
            val = all_reports[('pragmatic', 'instance_accuracy')]
            results.append((val, temp, nullcost, cost_function))
        maxval = np.max([x[0] for x in results])
        best = [x[1:] for x in results if x[0]==maxval]
        best = sorted(best, key=itemgetter(0))
        best = sorted(best, key=itemgetter(1))
        return dict(zip(['temperature', 'nullcost', 'cost_function'], best[0]))

class SpeakerRSA:
    def __init__(self, instance, temperature=1.0, cost_function=all_zero_costs, prior=None, nullcost=5.0):
        self.trial_id, self.target, self.msg, self.distractor_msgs, _, self.referents = instance              
        self.temperature = temperature
        self.cost_function = cost_function
        self.nullcost = nullcost
        self.costs = self.cost_function(self.distractor_msgs)
        self.costs = np.concatenate((self.costs, np.array([self.nullcost])))
        self.prior = prior
        if self.prior == None:
            self.prior = np.repeat(1.0/len(self.referents), len(self.referents))
        # Build the lexicon:
        self.lexicon = np.zeros((len(self.referents), len(self.distractor_msgs)+1))  
        self.build_lexicon()
        # Run RSA, which fills in these three matrices
        self.literal_speaker = None
        self.listener = None
        self.pragmatic_speaker = None
        self.rsa()

    def evaluate(self):
        results = {}
        for agent, agentname in ((self.literal_speaker, 'literal'), (self.pragmatic_speaker, 'pragmatic')):
            results[agentname] = self.evaluate_agent(agent)
        return results
         
    def evaluate_agent(self, agent):
        target_index = self.referents.index(self.target)
        target_row = agent[target_index]
        maxprob = np.max(target_row)
        max_msgs = [msg for j, msg in enumerate(self.distractor_msgs) if target_row[j]==maxprob]
        prediction = random.choice(max_msgs)
        results = {'prediction': prediction, 'actual': self.msg, 'evaluations':{}}
        for metric in (instance_accuracy, multiset_dice):            
            results['evaluations'][metric.__name__] = metric(prediction, self.msg)
        return results

    def build_lexicon(self):
        for i, ent in enumerate(self.referents):
            for j, msg in enumerate(self.distractor_msgs):
                if self.message_is_true_of_referent(msg, ent):
                    self.lexicon[i, j] = 1.0
        self.lexicon[:,-1] = 1.0
                                        
    def message_is_true_of_referent(self, msg_attrs, ent_attrs):
        for p in msg_attrs:
            if p not in ent_attrs:
                return False
        return True

    def rsa(self):
        self.literal_speaker = rownorm(np.exp(1.0 * (safelog(self.lexicon) - self.costs)))
        self.listener = rownorm(self.literal_speaker.T * self.prior)
        self.pragmatic_speaker = rownorm(np.exp(self.temperature * (safelog(self.listener.T) - self.costs)))


def domain_experiments():
    for dirname in ('furniture', ): #'people'):
        print "======================================================================"
        print dirname
        logfile = "logs/log_%s.pickle" % dirname
        mod = Experiment(glob('../TUNA/corpus/singular/%s/*.xml' % dirname), cv=5, logfile=logfile)
        mod.run()

def pooled_experiment(agentname='literal'):
    # Collapse across folds:
    results = defaultdict(lambda : defaultdict(list))
    for dirname in ('furniture', 'people'):
        log = pickle.load(file("logs/log_%s.pickle" % dirname))
        for d in log:
            fold = d['fold_index']
            acc = d[agentname]['evaluations']['instance_accuracy']
            dice = d[agentname]['evaluations']['multiset_dice']
            results[fold]['instance_accuracy'].append(acc)
            results[fold]['multiset_dice'].append(dice)
    # Means for the folds:
    pooled = defaultdict(dict)
    for fold, metric_vals in results.items():
        for metric, vals in metric_vals.items():
            pooled[metric][fold] = np.mean(vals)
    # Stats across the folds:
    runs = {}
    for metric, fold_dict in pooled.items():
        fold_vals = np.array(fold_dict.values())
        mu = np.mean(fold_vals)
        upper, lower = confidence_interval(fold_vals)
        print '%s mean %s: %0.03f (ci %0.03f, %0.03f)' % (agentname, metric, mu, upper, lower)
        runs[metric] = fold_vals
    return runs

if __name__ == '__main__':

    domain_experiments()
    #litruns = pooled_experiment(agentname='literal')
    #pragruns = pooled_experiment(agentname='pragmatic')

    #from scipy.stats import wilcoxon

    #for metric, litvals in litruns.items():
    #    pragvals = pragruns[metric]
    #    print metric, 'Wilcoxon T: %f; p = %f' % wilcoxon(litvals, pragvals)
        

    
