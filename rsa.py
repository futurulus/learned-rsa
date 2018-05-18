import numpy as np
import cPickle as pickle
from glob import glob
import random
import itertools
from collections import defaultdict
from operator import itemgetter
from tuna import Trial, TunaCorpus
from utils import rownorm, safelog, powerset, display_matrix,confidence_interval
from sklearn.cross_validation import KFold


class Experiment:
    def __init__(self, filenames=None, cv=5):        
        self.filenames = filenames
        self.cv = cv
        self.sem = set([])
        self.get_message_meanings()

    def run(self, filenames=None, temperature=1.0, nullcost=1.0):
        filenames = self.filenames if filenames == None else filenames
        all_reports = []
        corpus = TunaCorpus(filenames)
        for trial in corpus.iter_trials():
            mod = TunaRSA(trial, self.sem, temperature=temperature, nullcost=nullcost)
            mod.rsa()
            all_reports.append(mod.prediction_report())
        return all_reports

    def full_evaluation(self,  temperature=1.0, nullcost=1.0):
        all_reports = self.run(temperature=temperature, nullcost=nullcost)
        print self.summarize(all_reports)

    def crossvalidate(self):
        kf = KFold(n=len(self.filenames), n_folds=self.cv, shuffle=True)
        summaries = []
        temps = []
        for train_indices, test_indices in kf:
            train = [self.filenames[i] for i in train_indices]
            temp, nullcost = self.set_hyperparameters(train)            
            test = [self.filenames[i] for i in test_indices]
            all_reports = self.run(test, temperature=temp, nullcost=nullcost)
            summary = self.summarize(all_reports)
            summaries.append(summary)
            temps.append(temp)
            print 'Temp: %s; nullcost: %s; %s' % (temp, nullcost, str(summary))
        for name in ('Literal', 'Pragmatic', 'Speaker'):
            vals = np.array([s[name] for s in summaries])
            ci = confidence_interval(vals)
            print "%s mean accuracy: %0.2f (%0.2f-%0.2f)" % (name, vals.mean(), ci[0], ci[1])

    def set_hyperparameters(self, filenames):
        results = []
        temps = np.arange(0.1, 1.1, 0.1)
        nullcosts = np.arange(0.0, 2.0, 1.0)
        for temp, nullcost in itertools.product(temps, nullcosts):
            all_reports = self.run(filenames, temperature=temp, nullcost=nullcost)
            summary = self.summarize(all_reports)
            val = summary['Pragmatic']
            results.append((val, temp, nullcost))
        maxval = np.max([x[0] for x in results])
        best = [x[1:] for x in results if x[0]==maxval]
        best = sorted(best, key=itemgetter(0))
        best = sorted(best, key=itemgetter(1))
        return best[0]

    def summarize(self, reports):
        summary = {}
        for name in ('Literal', 'Pragmatic', 'Speaker'):
            num = sum([1.0 for report in reports if report[name]['Evaluation']])
            den = float(len(reports))
            acc = num/den
            summary[name] = acc
        return summary

    def get_message_meanings(self):
        corpus = TunaCorpus(self.filenames)
        for trial in corpus.iter_trials():
            entities = [[e] for e in trial.entities]
            if len(trial.targets) > 1:
                entities = get_non_identical_pairs(trial.entities)
            desc = trial.description.string_description
            attrs = sorted(set([str(a) for a in trial.description.attribute_set]))
            for ents in entities:
                if self.message_is_true_of_referent(attrs, ents):
                    self.sem.add((desc, tuple([e.id for e in ents])))

    def message_is_true_of_referent(self, attrs, ents):
        e_attrs = set([str(a) for e in ents for a in e.attributes])
        for p in attrs:
            if p not in e_attrs:
                return False
        return True

def get_non_identical_pairs(vals):
    n = len(vals)
    output = []
    for i in range(n-1):
        for j in range((i+1), n):
            output.append([vals[i], vals[j]])
    return output

######################################################################


class TunaRSA:
    def __init__(self, trial, sem, temperature=1.0, nullcost=5.0):
        self.trial = trial
        self.sem = sem
        self.temperature = temperature
        self.targets = trial.targets
        self.referents = [[r] for r in trial.entities]
        if len(self.targets) > 1:
            self.referents = get_non_identical_pairs(trial.entities)
        self.referent_ids = [tuple([r.id for r in refs]) for refs in self.referents]
        self.messages = sorted(set([d for d, ents in self.sem if ents in self.referent_ids]))
        self.actual_message = trial.description.string_description
        self.lexicon = np.zeros((len(self.messages)+1, len(self.referents)))
        for i, m in enumerate(self.messages):
            for j, refs in enumerate(self.referents):
                if (m, tuple([r.id for r in refs])) in sem:
                    self.lexicon[i, j] = 1.0
        self.lexicon[-1] = np.ones(len(self.referents))
        print 'Lex dims:', self.lexicon.shape
        #self.costs = np.array([len(m) for m in self.messages] + [nullcost])
        self.costs = np.repeat(0, len(self.messages)+1)
        self.costs[-1] = nullcost
        self.prior = np.repeat(1.0/len(self.referents), len(self.referents))

    def rsa(self):
        self.literal_listener = rownorm(self.lexicon * self.prior)
        self.speaker = rownorm(np.exp(self.temperature * (safelog(self.literal_listener.T) - self.costs)))
        self.final_listener = rownorm(self.speaker.T * self.prior)

    def predictions(self, listener='l1'):
        mat = self.final_listener if listener=='l1' else self.literal_listener
        if self.actual_message not in self.messages:
            return []
        i = self.messages.index(self.actual_message)
        row = mat[i]
        maxprob = np.max(row)
        return [self.referents[j] for j in range(len(self.referents)) if row[j] == maxprob]    

    def evaluate(self, preds):
        if not preds:
            return False
        referents = random.choice(preds)
        return set([r.type for r in referents]) == set(['target'])
    
    def speaker_predictions(self):
        i = self.referents.index([self.targets[0]])
        row = self.speaker[i]
        maxprob = np.max(row)
        return [self.messages[j] for j in range(len(self.messages)) if row[j] == maxprob]

    def speaker_evaluation(self, preds):
        if not preds:
            return False
        msg = random.choice(preds)
        return msg == self.actual_message

    def prediction_report(self):
        report = {'Target': [str(a) for a in self.trial.targets[0].attributes],
                  'Description': self.trial.description.string_description,
                  'Message': self.actual_message,
                  'Temperature': self.temperature,
                  'Literal': {},
                  'Pragmatic': {},
                  'Speaker': {}}
        literal = self.predictions(listener='l0')
        pragmatic = self.predictions(listener='l1')
        for name, preds in (('Literal', literal), ('Pragmatic', pragmatic)):
            report[name] = {'Evaluation': self.evaluate(preds), 'Predictions': []}
            for pred in preds:
                report[name]['Predictions'].append({'Types': [p.type for p in pred], 'Attributes': [str(a) for p in pred for a in p.attributes]})
        spk = self.speaker_predictions()
        report['Speaker'] = {'Evaluation': self.speaker_evaluation(spk)}
        return report


######################################################################

    
if __name__ == '__main__':

    #exp = Experiment(glob('../TUNA/corpus/singular/furniture/*.xml'))
    #exp.crossvalidate()

    #exp = Experiment(glob('../TUNA/corpus/singular/people/*.xml'))
    #exp.crossvalidate()

    #exp = Experiment(glob('../TUNA/corpus/plural/furniture/*.xml'))
    #exp.crossvalidate()

    exp = Experiment(glob('../TUNA/corpus/plural/people/*.xml'))
    exp.crossvalidate()
