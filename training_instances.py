import ast
from glob import glob
from tuna import *
from itertools import combinations
from utils import powerset
import rsa
import artificialdata


def get_singular_instances(filenames=glob("../TUNA/corpus/singular/people/*.xml")):
    corpus = TunaCorpus(filenames)
    D = []
    for trial in corpus.iter_trials():
        words = trial.description.unigrams()
        input_attrs = sorted(set([str(a) for a in trial.description.attribute_set]))
        target = trial.targets[0]
        attrs = [str(a) for a in target.attributes]
        domain = [[str(a) for a in e.attributes] for e in trial.entities]
        #if words == 'green desk last row'.split():
        #    print domain
        D.append((trial.id, words, attrs, domain, input_attrs))
    return D


def get_plural_instances(filenames=None):
    corpus = TunaCorpus(filenames)
    D = []
    for trial in corpus.iter_trials():
        words = trial.description.unigrams()
        input_attrs = sorted(set([str(a) for a in trial.description.attribute_set]))
        # In the plural domain, this always has len 2:
        targets = trial.targets
        # The target is the concatenation of both targets
        attrs = sorted([str(a) for target in targets for a in target.attributes])
        # The distractors are pairs of non-identical elements:
        domain = [(e1, e2) for e1, e2 in combinations(trial.entities, 2)]
        domain = [sorted([str(a) for a in e1.attributes + e2.attributes]) for e1, e2 in domain]
        D.append((trial.id, words, attrs, domain, input_attrs))
    return D


def get_generation_instances(filenames=None, max_length=None):
    corpus = TunaCorpus(filenames)
    D = []
    msg_map = {}
    for trial in corpus.iter_trials():
        target = sorted([str(a) for a in trial.targets[0].attributes])
        msg = sorted([str(a) for a in trial.description.attribute_set
                      if str(a) in target])
        distractors = [sorted(x) for x in powerset(target, minsize=1,
                                                   maxsize=max_length)]
        domain = [sorted([str(a) for a in e.attributes]) for e in trial.entities]
        D.append((trial.id, target, msg, distractors, target, domain))
    return D

def get_artificial_instances(filenames=None):
    return artificialdata.rsa_dataset(allow_ambiguities=True)


def is_relevant(message_attrs, y):
    return all(m in y for m in message_attrs)


def get_all_relevant_messages(domain, messages_attrs):
    domain = [set(y) for y in domain]
    return [message for message, attrs in messages_attrs
                    if any(is_relevant(attrs, y) for y in domain)]


def add_relevant_alts(dataset):
    messages_attrs = [(x, attrs) for _, x, _, _, attrs in dataset]
    
    return [(id, x, y, domain, attrs,
             get_all_relevant_messages(domain, messages_attrs))
            for id, x, y, domain, attrs in dataset]


def is_loc(y):
    if '[' in y:
        y = ast.literal_eval(y)
    return any('dimension' in a for a in y)


def filter_loc(dataset, loc=True, filenames=glob('../TUNA/corpus/*/*/*.xml')):
    """If `loc` is True, return all examples in `dataset` that come
    from trials with location references allowed (the "+LOC" condition).
    If `loc` is False, return all examples in `dataset` from the "-LOC"
    condition."""
    # XXX: we shouldn't need to read the dataset twice. Should probably
    # keep the full trial so we don't need to keep adding elements to
    # the tuple or doing hacks like this.
    plusloc_trials = set()
    for trial in TunaCorpus(filenames=filenames).iter_trials():
        if trial.condition == '+LOC':
            plusloc_trials.add(trial.id)
        else:
            assert trial.condition == '-LOC'
    
    # id, x, y, domain, attrs[, alts]
    #  0  1  2       3      4      5
    return [d for d in dataset
            if (d[0] in plusloc_trials) == loc]
