"""Unit tests for sgd_lsl (that aren't suitable for doctest)."""

import unittest
import numpy as np
from collections import defaultdict
from glob import glob
from tuna import TunaCorpus
from tokenizers import basic_unigram_tokenizer
from learning_rsa import LSLTrainer, FeatureVectorizer


def phi_unigrams(phrase, answer):
    feats_dict = {'%s->%s' % (w, r): 1.0
                  for w in phrase.split() for r in answer}
    return defaultdict(float, feats_dict)


def get_ids(ent_set):
    return tuple(e.id for e in ent_set)


def find_matching_messages(sem, distractors):
    return [basic_unigram_tokenizer(message)
            for message, refids in sem.iteritems()
            if any(get_ids(d) in refids for d in distractors)]


def sem_to_message_sets(sem, ents):
    merged_sem = defaultdict(set)
    for message, refids in sem:
        merged_sem[message].update(set([refids]))
    for k, v in merged_sem.iteritems():
        print "%s: %s" % (k, v)

    return [find_matching_messages(merged_sem, distractor_set)
            for distractor_set in ents]


class LSLTest(unittest.TestCase):
    def test_add_relevant_alts(self):
        from training_instances import add_relevant_alts

        dataset = [
            ('a', ['red', 'chair'], ['RED', 'SMALL', 'CHAIR'],         # id, x, y
             [['BLUE', 'LARGE', 'TABLE'], ['RED', 'SMALL', 'CHAIR']],  # domain
             ['RED', 'CHAIR']),                                        # attrs

            ('b', ['blue', 'desk'], ['BLUE', 'LARGE', 'TABLE'],
             [['BLUE', 'LARGE', 'TABLE'], ['RED', 'LARGE', 'TABLE']],
             ['BLUE', 'TABLE']),

            ('c', ['big', 'desk'], ['RED', 'LARGE', 'TABLE'],
             [['RED', 'LARGE', 'TABLE'], ['RED', 'SMALL', 'CHAIR']],
             ['LARGE', 'TABLE']),
        ]

        augmented = add_relevant_alts(dataset)

        ms_actual = [a for _, _, _, _, _, a in augmented]

        ms_expected = [
            [['red', 'chair'], ['blue', 'desk'], ['big', 'desk'],],
            [['blue', 'desk'], ['big', 'desk'],],
            [['red', 'chair'], ['big', 'desk'],],
        ]

        for expected, actual in zip(ms_expected, ms_actual):
            self.assertItemsEqual(expected, actual)

    def test_simple_gradient(self):
        MESSAGES = ['hello world', 'hello alice', 'hello']
        REFERENTS = [(), ('alice',), ('bob',), ('alice', 'bob')]

        weights = defaultdict(float)

        vec = FeatureVectorizer(phi_unigrams)

        trainer = LSLTrainer(phi=phi_unigrams)
        trainer.vectorizer = vec
        actual = trainer.gradient('hello world', MESSAGES,
                                  ('alice', 'bob'), REFERENTS,
                                  weights,
                                  verbose=0)

        expected = {
            'world->bob': 0.33333333333333337,
            'alice->alice': -0.16666666666666666,
            'alice->bob': -0.16666666666666666,
            'world->alice': 0.33333333333333337
        }
        
        self.assertEqual(expected, actual)

    @unittest.skip('sem is buggy')
    def test_relevant_messages_matches_rsa_sem(self):
        from training_instances import get_plural_instances, add_relevant_alts
        import rsa
        
        FILENAMES = glob('../TUNA/corpus/plural/furniture/*.xml')
        dataset = add_relevant_alts(get_plural_instances(FILENAMES))
        print 'dataset:'
        for id, x, y_true, domain, attrs, alts in dataset:
            print (id, x, y_true, attrs)
            for y in domain:
                print '  %s,' % (y,)
        
        ms_actual = [a for _, _, _, _, _, a in dataset]
        print
        print 'actual:'
        for a in ms_actual:
            print a

        ex = rsa.Experiment(FILENAMES)
        ex.get_message_meanings(include_all=True)
        
        c = TunaCorpus(FILENAMES)
        ents = []
        print
        for trial in c.iter_trials():
            print get_ids(trial.entities)
            if len(trial.targets) > 1:
                ents.append(tuple(rsa.get_non_identical_pairs(trial.entities)))
            else:
                ents.append(tuple((e,) for e in trial.entities))
        print
        print 'ents:'
        for e in ents:
            print list(get_ids(t) for t in e)
        
        ms_expected = sem_to_message_sets(ex.sem, ents)
        print
        print 'expected:'
        for a in ms_expected:
            print a
        
        self.maxDiff = None
        for expected, actual in zip(ms_expected, ms_actual):
            self.assertItemsEqual(expected, actual)
        

if __name__ == '__main__':
    unittest.main()