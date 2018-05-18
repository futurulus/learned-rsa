from collections import Counter
from itertools import product, combinations


def cross_product_features(x, y):
    return Counter(['%s|%s' % (a, b) for a, b in product(x, y)])


def null_features(x, y):
    """This is equivalent to random guessing because of the way SGD breaks ties."""
    return Counter([])


def attribute_type_features(x, y):
    return Counter(['%s:*' % a.split(':')[0] for a in y])


def attribute_pair_features(x, y):
    mentioned = set(attribute_type_features(x, y).keys())
    not_mentioned = set(attribute_type_features(x, x).keys()) - mentioned

    return Counter(['%s+%s' % (a, b) for a, b in combinations(mentioned, 2)] +
                   ['%s+NO_%s' % (a, b) for a, b in product(mentioned, not_mentioned)] +
                   ['NO_%s+NO_%s' % (a, b) for a, b in combinations(not_mentioned, 2)])


def attribute_count_features(x, y):
    return Counter(['#:%d' % len(y)])


def falsehood_features(x, y):
    return Counter(['[FALSEHOOD]' for a in y if a not in x])


def phi(feat_names):
    def features(x, y):
        f = Counter()
        for name in feat_names:
            f.update(FEATURES[name](x, y))
        return f
    return features


FEATURES = {
    'null': null_features,
    'cross_product': cross_product_features,
    'attr_type': attribute_type_features,
    'attr_count': attribute_count_features,
    'attr_pair': attribute_pair_features,
    'falsehood': falsehood_features,
}