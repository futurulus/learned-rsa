from itertools import product
from collections import defaultdict, Counter
import random
import numpy as np
from utils import rownorm
from learning import LiteralTrainer
from featurefunctions import cross_product_features
import config

parser = config.get_options_parser()
parser.add_argument('--ambiguities', type=config.boolean, default=False)
parser.add_argument('--mat_size', type=int, default=2)


def rsa(mat):
    return rownorm(rownorm(rownorm(mat).T).T)


def all_matrices(nrow=2, ncol=2):
    for x in product((0.0,1.0), repeat=nrow*ncol):
        mat = np.array(x).reshape((nrow, ncol))
        if not 0.0 in mat.sum(axis=0) and 0.0 not in mat.sum(axis=1):
            yield mat


def rsa_dataset(nrow=2, ncol=2, allow_ambiguities=True):
    D = []
    for mat_index, mat in enumerate(all_matrices(nrow=nrow, ncol=ncol)):
        # The model provides the training signal:
        mod = rsa(mat) 
        for i, msg in enumerate(mod):
            # The target is a random draw from the best guesses for the model:
            best_guesses = [j for j, val in enumerate(msg) if val==msg.max()]
            if len(best_guesses)==1 or allow_ambiguities:
                target = random.choice(best_guesses)                                   
                # Our usual format for training -- importantly, the objects in here are 
                # the truth conditions, not the possibly pragmatic vectors of mod.
                # training_instance = (mat_index, i, list(mat[:,target]), [list(y) for y in mat.T], None)
                training_instance = (mat_index, list(mod[i]), target, range(mod.shape[1]), None)
                # training_instance = (mat_index, list(mat[i]), list(mat[:,target]), [list(y) for y in mat.T], None)
                # training_instance = (mat_index, list(mat[i]), list(mat[:,target]), [list(y) for y in mat.T], None)
                D.append(training_instance)
    return D


def artificial_features(x, y):
    x_ind = range(len(x))
    y_ind = range(len(y))
    return Counter([str((i,j)) for i, j in product(x_ind, y_ind) if x[i]==1.0 and y[j]==1.0])


def exact_artificial_features(x, y):
    x_ind = range(len(x))
    y_ind = range(len(y))    
    return Counter([str((str(x),str(y))) for i, j in product(x_ind, y_ind) if x[i]==max(x) and y[j]==max(y)])


def index_artificial_features(x, y):
    return Counter([str((str(x), str(y)))])


if __name__ == '__main__':
    options = config.options()
    D = rsa_dataset(nrow=options.mat_size, ncol=options.mat_size,
                    allow_ambiguities=options.ambiguities)    
    trainer = LiteralTrainer(data=D, train_percentage=0.0, phi=index_artificial_features, T=10)
    trainer.train_test_evaluation_report(verbose=0)
    
