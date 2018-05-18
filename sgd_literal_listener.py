from glob import glob
from tuna import *
from collections import defaultdict
from itertools import product
from learning import LiteralTrainer
from featurefunctions import null_features, cross_product_features, FEATURES
import featurefunctions
import training_instances as inst
import numpy as np
import config


parser = config.get_options_parser()
parser.add_argument('--data_dir', type=str, default='singular/furniture')
parser.add_argument('--features', choices=FEATURES.keys(), metavar='FEAT_NAME',
                    nargs='*', default=['cross_product'])
parser.add_argument('--sgd_eta', type=float, default=0.1)
parser.add_argument('--cv', type=int, default=10)
parser.add_argument('--random_splits', type=config.boolean, default=False)
parser.add_argument('--train_percentage', type=float, default=0.8)
parser.add_argument('--sgd_max_iters', type=int, default=50)
parser.add_argument('--l2_coeff', type=float, default=0.0)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--generation', type=config.boolean, default=False)


def check_target_counts():
    for dirname in ('plural/furniture', 'plural/people'):
        filenames = glob("../TUNA/corpus/%s/*.xml" % dirname)
        corpus = TunaCorpus(filenames)
        counts = defaultdict(int)
        for trial in corpus.iter_trials():
            counts[len(trial.targets)] += 1
        print dirname, counts


def evaluate_all(
        dirnames=('singular/furniture', 'singular/people'),
        instance_function=inst.get_singular_instances,
        cv=10,
        train_percentage=0.8,
        random_splits=False,
        T=50,
        features=FEATURES.values(),
        eta=0.1,
        l2_coeff=0.0,
        verbose=0,
        typ='listener'):
    for dirname in dirnames:
        filenames = glob("../TUNA/corpus/%s/*.xml" % dirname)
        data = instance_function(filenames=filenames)
        for phi in features:
            trainer = LiteralTrainer(data=data, dirname=dirname, phi=phi,
                                     eta=eta, l2_coeff=l2_coeff,
                                     random_splits=random_splits,
                                     train_percentage=train_percentage,
                                     T=T, cv=cv, typ=typ)
            trainer.cv_evaluation_report(verbose=verbose)


def main():
    options = config.options()

    instance_function = (inst.get_generation_instances
                         if options.generation else
                         inst.get_plural_instances
                         if 'plural' in options.data_dir else
                         inst.get_singular_instances)

    evaluate_all(dirnames=(options.data_dir,),
                 features=(featurefunctions.phi(options.features),),
                 instance_function=instance_function,
                 cv=options.cv,
                 random_splits=options.random_splits,
                 T=options.sgd_max_iters,
                 l2_coeff=options.l2_coeff,
                 eta=options.sgd_eta,
                 verbose=options.verbose,
                 typ='speaker' if options.generation else 'listener')

    #evaluate_all(cv=5, verbose=1)
    #evaluate_all(cv=5, verbose=1, dirnames=('plural/furniture', 'plural/people'))
    #evaluate_all(cv=10, l2_coeff=0.0, verbose=1)


if __name__ == '__main__':
    main()
