from glob import glob
from learning import LiteralTrainer
from featurefunctions import null_features, cross_product_features
from metrics import accuracy, mean_multiset_dice, max_multiset_dice
from training_instances import get_generation_instances


def evaluate_all(cv=5, eta=0.1, verbose=1, dirnames=('singular/furniture', 'singular/people')):
    for dirname in dirnames:
        filenames = glob("../TUNA/corpus/%s/*.xml" % dirname)
        data = get_generation_instances(filenames=filenames)
        for phi in (cross_product_features,):
            trainer = LiteralTrainer(data=data, dirname=dirname, phi=phi, eta=eta, cv=cv, typ="speaker",
                                     metrics=[accuracy,max_multiset_dice,mean_multiset_dice])
            trainer.cv_evaluation_report(verbose=verbose)            
            

if __name__ == '__main__':

   #evaluate_all()

   evaluate_all(cv=5, eta=0.1, verbose=1, dirnames=('singular/people',))
