#!/usr/bin/env python

import sys
from math import sqrt
from collections import defaultdict
from operator import itemgetter
import numpy as np
import json
import traceback
import datetime
from metrics import accuracy
from utils import confidence_interval
from sklearn.cross_validation import KFold, ShuffleSplit
import config
import timing


random = np.random.RandomState(np.uint32(hash('train')))

def score(x=None, y=None, phi=None, w=None):
    return sum(w[f]*count for f, count in phi(x, y).items())

def cost(y, y_prime):
    return 0.0 if y == y_prime else 1.0

class LiteralTrainer(object):
    def __init__(self,
            data=None,
            cv=10,
            random_splits=False,
            train_percentage=0.8,
            phi=None,
            T=100,
            eta=0.1,
            l2_coeff=0.0,
            metrics=[accuracy],
            typ=None,
            dirname=None,
            epsilon=sys.float_info.epsilon):
        self.data = data
        self.cv = cv
        self.random_splits = random_splits
        self.train_percentage = train_percentage
        self.phi = phi
        self.T = T
        self.eta = eta
        self.l2_coeff = l2_coeff
        self.metrics = metrics
        self.epsilon = epsilon
        self.typ = typ
        self.dirname = dirname

    def predict(self, x=None, w=None, phi=None,
                messages=None, distractors=None):
        scores = [(score(x, y_prime, self.phi, w), y_prime) for y_prime in distractors]
        # Get the maximal score:
        max_score = sorted(scores)[-1][0]
        # Get all the candidates with the max score and choose one randomly:
        y_hats = [y_alt for s, y_alt in scores if s == max_score]
        y_hat = y_hats[random.choice(range(len(y_hats)))]
        if len(y_hats) > 1:
            print 'Guessed: %s -> %s (of %d)' % (x, y_hat, len(y_hats))
        return y_hat

    def SGD(self, D=None, verbose=0, l2_coeff=None):
        l2_coeff = self.l2_coeff if l2_coeff == None else l2_coeff
        weights = defaultdict(float)
        for iteration in range(self.T):
            random.shuffle(D)
            error = 0.0
            update_mag = 0.0
            for d in D:
                id_, x, y, distractors = d[:4]
                
                # Get all (score, y') pairs:
                scores = [(score(x, y_alt, self.phi, weights)+cost(y, y_alt), y_alt) for y_alt in distractors]
                # Get the maximal score:
                max_score = sorted(scores)[-1][0]
                # Get all the candidates with the max score and choose one randomly:
                y_tildes = [y_alt for s, y_alt in scores if s == max_score]
                y_tilde = y_tildes[random.choice(range(len(y_tildes)))]
                # Error
                derr = max_score - score(x, y, self.phi, weights)
                # print 'derr=%f %s -> %s (guess: %s, of %d)' % (derr, x, y, y_tilde, len(y_tildes))
                error += derr
                # Featurized:
                actual_rep = self.phi(x, y)
                predicted_rep = self.phi(x, y_tilde)                
                # Gradients:
                grad = defaultdict(float)
                for f in set(actual_rep.keys() + predicted_rep.keys()):
                    grad[f] = actual_rep[f] - predicted_rep[f]
                # Regularization:
                for f, w in weights.items():
                    grad[f] -= l2_coeff * w
                # Update:
                for f in grad.keys():
                    dw = self.eta * grad[f]
                    weights[f] += dw
                    update_mag += dw ** 2
            if verbose: 
                print 'Error: %f' % error
                print 'Weight update magnitude: %f' % update_mag
            if error <= self.epsilon:
                if verbose:
                    print "Terminating after %s iterations; error is minimized." % iteration
                return (weights, error, iteration, None)
        if verbose:
            print "Terminating after max iterations reached; error is %0.02f." % error
        return (weights, error, iteration, None)

    def cv_evaluation_report(self, verbose=0):
        all_results = self.evaluate_cv(data=self.data, verbose=verbose)
        split_info = "Cross-validation folds: %s" % self.cv
        self.evaluation_report(all_results, verbose=verbose, split_info=split_info)

    def evaluate_cv(self, data=None, verbose=0):
        all_results = []
        if self.random_splits:
            splits = ShuffleSplit(n=len(data), n_iter=self.cv,
                                  test_size=1 - self.train_percentage, random_state=random)
        else:
            splits = KFold(n=len(data), n_folds=self.cv, shuffle=True, random_state=random)
        if verbose <= 1:
            timing.set_resolution(datetime.timedelta(minutes=5))
        timing.start_task('Train split' if self.random_splits else 'CV fold', self.cv)
        for eval_num, (train_indices, test_indices) in enumerate(splits):
            timing.progress(eval_num)
            train = [self.data[i] for i in train_indices]
            test = [self.data[i] for i in test_indices]
            all_results.append(self.evaluate(train=train, test=test,
                                             eval_num=eval_num, verbose=verbose))
        timing.end_task()
        return all_results

    def evaluate_train_test(self, eval_num=0, verbose=0):
        """verbose=0 for no report; 1 for a final report; 2 for a final report and weight report"""
        # Train-test split:
        train, test = self.train_test_split()
        return self.evaluate(train=train, test=test, eval_num=eval_num, verbose=verbose)

    def train_test_evaluation_report(self, verbose=0, trials=10):
        all_results = [self.evaluate_train_test(eval_num=i, verbose=verbose) for i in range(trials)]
        split_info = "Train percentage: %s" % self.train_percentage
        split_info += "\nTrials: %s" % len(all_results)
        self.evaluation_report(all_results, verbose=verbose, split_info=split_info)

    def train_test_split(self):
        train, test = None, None
        if self.train_percentage > 0.0: # where 0, no split:
            random.shuffle(self.data)
            train_size = int(round(len(self.data)*self.train_percentage, 0))
            train = self.data[ : train_size]
            test = self.data[train_size: ]
        else:
            print 'Warning: training and testing on the same data!'
            train = self.data
            test = self.data
        return (train, test)

    def evaluate(self, train=None, test=None, verbose=0, eval_num=0, params=None): #np.arange(0.0, 2.0, 0.1)):
        coef, weights, error, iterations = None, None, None, None
        if params != None:
            coef, weights, error, iterations = self.grid_search_train(train, params=params)
        else:
            coef = self.l2_coeff
            weights, error, iterations, messages = self.SGD(
                D=train, l2_coeff=self.l2_coeff, verbose=verbose)
        # Optionally view weights:
        if verbose == 2:
            for key, val in sorted(weights.items(), key=itemgetter(1, 0), reverse=True):
                if val != 0.0:
                    print key, val
        config.dump(weights, 'params.%s.json' % eval_num)
        # Evaluation:
        predictions = []
        for i, d in enumerate(test):
            (_, x, _, distractors) = d[:4]

            if verbose >= 2:
                print 'Evaluating %d of %d' % (i, len(test))
            predictions.append(self.predict(x=x, w=weights,
                                            messages=messages, distractors=distractors))
        gold = [d[2] for d in test]  # id, x, y, ...
        with config.open('predictions.%s.jsons' % eval_num, 'w') as outfile:
            for d, prediction in zip(test, predictions):
                (id, x, y) = d[:3]

                report = {'id': str(id), 'input': str(x),
                          'gold': str(y), 'prediction': str(prediction)}
                json.dump(report, outfile)
                if verbose >= 2 and y != prediction:
                    print 'Wrong: %s' % str(report)
                outfile.write('\n')
        results = {'iterations':iterations, 'error': error, 'evaluations':{}, 'l2_coeff': coef}
        for metric in self.metrics:
            evaluation = metric(gold, predictions)
            results['evaluations'][metric.__name__] = evaluation
            if verbose:
                print "%s: %0.03f" % (metric.__name__, evaluation)
        return results

    def grid_search_train(self, train, metric=None, params=None):
        """Add regularizer and fill this in"""
        metric = self.metrics[0] if metric == None else metric
        all_evaluations = []
        for coef in params:
            kf = KFold(n=len(train), n_folds=self.cv, shuffle=True)
            evaluations = []
            for train_indices, test_indices in kf:
                fold_train = [train[i] for i in train_indices]
                fold_test = [train[i] for i in test_indices]
                weights, error, iterations = self.SGD(D=fold_train, l2_coeff=coef)
                predictions = [self.predict(x=x, w=weights, distractors=distractors) for x, _, distractors in fold_test]
                gold = [y for _, y, _ in fold_test]
                evaluations.append(metric(gold, predictions))
            all_evaluations.append((coef, np.mean(evaluations)))
        weights, error, iterations = self.SGD(D=train, l2_coeff=coef)
        return (coef, weights, error, iterations)

    def evaluation_report(self, all_results, verbose=0, split_info=None):
        errors = np.array([d['error'] for d in all_results])
        iterations = np.array([d['iterations'] for d in all_results])
        print "======================================================================"
        print "Type: %s" % self.typ
        print "Domain: %s" % self.dirname
        print "Features: %s" % self.phi.__name__
        print split_info
        print "Learning rate: %s" % self.eta
        print "L2 coefs:", [r['l2_coeff'] for r in all_results]
        print "Mean iterations to convergence:  %0.3f (+/- %0.3f)" % (iterations.mean(), iterations.std()*2)
        for metric in self.metrics:
            vals = np.array([d['evaluations'][metric.__name__] for d in all_results])
            ci = confidence_interval(vals)
            print "Mean %s: %0.3f (%.3f--%.3f)" % (metric.__name__, vals.mean(), ci[0], ci[1])

