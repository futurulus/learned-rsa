from glob import glob
from collections import Counter, defaultdict
from operator import itemgetter
import numpy as np
import json
import traceback

from tuna import *
import featurefunctions
from featurefunctions import FEATURES
from learning_rsa import LSLTrainer
from metrics import accuracy, mean_multiset_dice
import training_instances as inst
import artificialdata
import config


parser = config.get_options_parser()

parser.add_argument('--data_dir', type=str, default='singular/furniture')
parser.add_argument('--generation', type=config.boolean, default=False)
parser.add_argument('--filter_loc', type=config.boolean, default=None)

parser.add_argument('--artificial', type=config.boolean, default=False)
parser.add_argument('--ambiguities', type=config.boolean, default=False)
parser.add_argument('--mat_size', type=int, default=2)

parser.add_argument('--random_splits', type=config.boolean, default=False)
parser.add_argument('--train_percentage', type=float, default=None)
parser.add_argument('--cv', type=int, default=10)

parser.add_argument('--features', choices=FEATURES.keys(), metavar='FEAT_NAME',
                    nargs='*', default=['cross_product'])

parser.add_argument('--max_gen_length', type=int, default=None)
parser.add_argument('--samples_x', type=int, default=None)
parser.add_argument('--samples_y', type=int, default=None)
parser.add_argument('--null_message', type=config.boolean, default=False)
parser.add_argument('--only_relevant_alts', type=config.boolean, default=False)
parser.add_argument('--only_local_alts', type=config.boolean, default=False)

parser.add_argument('--sgd_max_iters', type=int, default=50)
parser.add_argument('--sgd_eta', type=float, default=0.01)
parser.add_argument('--sgd_use_adagrad', type=config.boolean, default=False)
parser.add_argument('--l2_coeff', type=float, default=0.0)

parser.add_argument('--verbose', type=int, default=0)

# obsolete?
parser.add_argument('--cache_featurizations', type=config.boolean, default=True)
parser.add_argument('--literal', type=config.boolean, default=False)


def evaluate(options,
             instance_function=inst.get_singular_instances):
    dirname = options.data_dir
    filenames = glob("../TUNA/corpus/%s/*.xml" % dirname)
    data = instance_function(filenames=filenames)
    
    if options.filter_loc is not None:
        data = inst.filter_loc(data, options.filter_loc, filenames=filenames)
    
    metrics = ([accuracy, mean_multiset_dice]
               if options.generation else
               [accuracy])
    phi = (artificialdata.index_artificial_features
           if options.artificial else
           featurefunctions.phi(options.features))

    trainer = LSLTrainer(data=data, dirname=dirname, phi=phi,
                         metrics=metrics,
                         use_adagrad=options.sgd_use_adagrad,
                         train_percentage=options.train_percentage,
                         random_splits=options.random_splits,
                         eta=options.sgd_eta,
                         l2_coeff=options.l2_coeff,
                         T=options.sgd_max_iters,
                         samples_x=options.samples_x,
                         samples_y=options.samples_y,
                         only_relevant_alts=options.only_relevant_alts,
                         only_local_alts=options.only_local_alts,
                         null_message=options.null_message,
                         cv=options.cv,
                         typ="listener")
    if (not options.random_splits) and (options.train_percentage is not None):
        trainer.train_test_evaluation_report(verbose=options.verbose)
    else:
        trainer.cv_evaluation_report(verbose=options.verbose)


def main():
    options = config.options()

    artificial_inst = lambda filenames: artificialdata.rsa_dataset(nrow=options.mat_size,
                                                                   ncol=options.mat_size,
                                                                   allow_ambiguities=options.ambiguities)
    instance_function = (artificial_inst
                         if options.artificial else
                         (lambda filenames: inst.get_generation_instances(filenames,
                                                options.max_gen_length))
                         if options.generation else
                         inst.get_plural_instances
                         if 'plural' in options.data_dir else
                         inst.get_singular_instances)

    evaluate(options, instance_function=instance_function)

'''
    filenames = glob("../TUNA/corpus/%s/*.xml" % options.data_dir)
    featname, phi, loss_grad = (('Literal listener', cross_product_features, log_loss_grad)
                                if options.literal else
                                ('RSA', cross_product_features, rsa_grad))
    accs = np.array([evaluate(filenames=filenames, phi=phi, loss_grad=loss_grad,
                              eval_num=i, options=options)
                     for i in range(options.evaluate_reps)])
    print 'Finished %d random 80/20 splits using %s on %s' % \
    (options.evaluate_reps, featname, options.data_dir)
    print 'accuracy mean: %0.3f' % accs.mean()
    print 'accuracy std: %0.3f' % accs.std()
'''

if __name__ == '__main__':
    main()
