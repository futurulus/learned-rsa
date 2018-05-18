import sys
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.misc import logsumexp
from scipy import sparse

from visualization import print_matrix, plot_matrix
import training_instances as inst
from learning import LiteralTrainer, score, cost
import timing


random = np.random.RandomState(np.uint32(hash('train')))

def find_or_add(actual, alts):
    try:
        idx = alts.index(actual)
    except ValueError:
        idx = len(alts)
        alts = alts + [actual]
    return idx, alts


def log_loss_grad(vectorizer, x_actual, x_alts, y_actual, y_alts, scorer, w, verbose=0):
    """Return the gradient of the standard log-linear loss function
    for a prediction `y_predicted` given the gold answer `y_actual`,
    as a dict mapping feature names to gradient values.
    
    Should be obsolete (use LiteralTrainer instead)."""
    featurizations_tensor, weights, names = featurize_all(
        vectorizer.phi, y_alts, [x_actual], w
    )
    #weights = vectorizer.vectorize(w).transpose()
    scores = featurizations_tensor.dot(weights).toarray().flatten()
    # Get the maximal score:
    max_score = sorted(scores)[-1]
    # Get all the candidates with the max score and chose one randomly:
    y_tildes = [y_alt for s, y_alt in zip(scores, y_alts) if s == max_score]
    y_tilde = y_tildes[random.choice(range(len(y_tildes)))]
    phi_actual = vectorizer.phi(x_actual, y_actual)
    phi_predicted = vectorizer.phi(x_actual, y_tilde)
    grad = Counter(dict(phi_actual))
    grad.subtract(phi_predicted)
    return grad


class LSLTrainer(LiteralTrainer):
    def __init__(self, use_adagrad=True,
                 samples_x=None, samples_y=None,
                 only_relevant_alts=False,
                 only_local_alts=False,
                 null_message=False,
                 *args, **kwargs):
        super(LSLTrainer, self).__init__(*args, **kwargs)
        self.use_adagrad = use_adagrad
        self.samples_x = samples_x
        self.samples_y = samples_y
        self.null_message = null_message
        self.only_relevant_alts = only_relevant_alts
        self.only_local_alts = only_local_alts

    def predict(self, x, w, messages, distractors):
        messages = self.sample(messages, self.samples_x)
        # no sampling of distractors, so we don't limit our accuracy

        x_index, messages = find_or_add(x, messages)
        if self.null_message:
            messages = messages + ['']

        literal_scores = log_softmax(np.array([[score(x_alt, y, self.phi, w) for y in distractors]
                                               for x_alt in messages]),
                                     axis=1)
        speaker_scores = log_softmax(literal_scores, axis=0)
        listener_scores = log_softmax(speaker_scores, axis=1)
        
        #plot_matrix(listener_scores - literal_scores, messages, distractors)

        listener_scores = listener_scores[x_index, :]

        scores = zip(listener_scores, distractors)

        # Get the maximal score:
        max_score = sorted(scores)[-1][0]
        # Get all the candidates with the max score and choose one randomly:
        y_hats = [y for s, y in scores if s == max_score]
        return y_hats[random.choice(range(len(y_hats)))]

    def sample(self, alts, num_to_sample):
        """Choose `num_to_sample` elements randomly without replacement from `alts`. If
        `num_to_sample` is None, non-positive, or greater than the size of `alts`, return
        `alts`."""
        if not num_to_sample or not (0 < num_to_sample < len(alts)):
            return alts
        return [alts[i] for i in random.choice(range(len(alts)), num_to_sample, replace=False)]

    def gradient(self, x_actual, x_alts, y_actual, y_alts, w, verbose=0):
        """Gradient of the RSA L(S(L)) model."""
        x_alts = self.sample(x_alts, self.samples_x)
        y_alts = self.sample(y_alts, self.samples_y)
        if self.null_message:
            x_alts = x_alts + ['']
        
        x_index, x_alts = find_or_add(x_actual, x_alts)
        y_index, y_alts = find_or_add(y_actual, y_alts)
        if verbose >= 2:
            print('x_index = %d; len(x_alts) = %d' % (x_index, len(x_alts)))

        featurizations_tensor, weights, names = featurize_all(self.vectorizer.phi,
                                                              y_alts, x_alts, w)
        if verbose >= 2:
            print('featurizations_tensor.shape:')
            print(len(featurizations_tensor), featurizations_tensor[0].shape)
        if verbose >= 3:
            print_matrix(featurizations_tensor)

        #weights = self.vectorizer.vectorize(w).transpose()
        weights = weights.transpose()
        if verbose >= 2:
            print('weights.shape:')
            print(weights.shape)
        if verbose >= 3:
            self.vectorizer.print_features(weights)

        try:
            literal_scores = log_softmax(np.hstack([feats_y.dot(weights).toarray()
                                                    for feats_y in featurizations_tensor]),
                                         axis=1)
        except ValueError:
            print 'featurizations_tensor[0].shape: %s' % (featurizations_tensor[0].shape,)
            print 'feats_y.shape: %s' % (feats_y.shape,)
            print 'weights.shape: %s' % (weights.shape,)
            print x_actual
            print y_actual
            raise
        if verbose >= 3:
            print('Literal scores:')
            print_matrix(literal_scores)

        literal_probs = sparse.csr_matrix(np.exp(literal_scores))
        literal_expected_phi = np.sum((featurizations_tensor[yi].multiply(literal_probs[:, yi])
                                       for yi in xrange(len(featurizations_tensor))), axis=2)
        if verbose >= 3:
            print('')
            print('Literal-expected phi:')
            print_matrix(literal_expected_phi)
        literal_expected_phi = sparse.csr_matrix(literal_expected_phi)

        grad_literal = [featurizations_tensor[yi] - literal_expected_phi
                        for yi in xrange(len(featurizations_tensor))]
        if verbose >= 3:
            print('')
            print('Gradient of literal:')
            print_matrix(grad_literal)

        speaker_scores = log_softmax(literal_scores, axis=0)
        if verbose >= 3:
            print('')
            print('Speaker scores:')
            print_matrix(speaker_scores)

        speaker_probs = sparse.csr_matrix(np.exp(speaker_scores))
        speaker_expected_grad_literal = np.vstack(((grad_literal[yi].multiply(
                                                    speaker_probs[:, yi])).sum(axis=0)
                                                   for yi in xrange(len(featurizations_tensor))))
        speaker_expected_grad_literal = np.array(speaker_expected_grad_literal)
        if verbose >= 3:
            print('')
            print('Speaker-expected gradient of literal:')
            print_matrix(speaker_expected_grad_literal)

        grad_speaker = np.vstack((grad_literal[yi][x_index, :].toarray()
                                  for yi in xrange(len(featurizations_tensor)))) - \
                       speaker_expected_grad_literal
        if verbose >= 3:
            print('')
            print('Gradient of speaker:')
            print_matrix(grad_speaker)

        listener_scores = log_softmax(speaker_scores[x_index, :])
        if verbose >= 3:
            print('')
            print('Listener scores:')
            print_matrix(listener_scores)

        listener_expected_grad_speaker = (grad_speaker *
                                          np.exp(listener_scores)[:, np.newaxis]).sum(axis=0)
        if verbose >= 3:
            print('')
            print('Listener-expected gradient of speaker:')
            self.vectorizer.print_features(listener_expected_grad_speaker)

        grad = grad_speaker[y_index, :] - listener_expected_grad_speaker
        if verbose >= 3:
            print('')
            print('Gradient of listener (final gradient):')
            self.vectorizer.print_features(grad)

        return unvectorize(grad, names)

    def SGD(self, D=None, l2_coeff=None, verbose=0):
        """Implements stochatic (sub)gradient descent. `D` should be an iterable
        of `(id, x, y, domain, attrs)` tuples, where domain is a list of possible
        outputs (`y in domain` should be `True`) and attrs is the list of object
        properties expressed by `x`. `messages` should be a list of possible inputs."""

        if verbose >= 1:
            print 'Training with eta=%f, l2_coeff=%f, use_adagrad=%s' % \
                (self.eta, self.l2_coeff, self.use_adagrad)

        if self.only_relevant_alts:
            D = inst.add_relevant_alts(D)
        elif not self.only_local_alts:
            # messages is the set of utterances observed in training, as a proxy for
            # the set of all possible utterances.
            messages = [d[1] for d in D]

        l2_coeff = self.l2_coeff if l2_coeff == None else l2_coeff

        self.vectorizer = FeatureVectorizer(phi=self.phi, verbose=verbose)

        weights = defaultdict(float)
        adagrad = defaultdict(lambda: 0.0)
        timing.start_task('Iteration', self.T)
        for iteration in range(self.T):
            timing.progress(iteration)
            #if verbose:
            #    print('Iteration %d of %d' % (iteration, self.T))
            random.shuffle(D)
            error = 0.0
            update_mag = 0.0
            timing.start_task('Example', len(D))
            for i, d in enumerate(D):
                timing.progress(i)
                if self.only_relevant_alts or self.only_local_alts:
                    (id_, x, y, domain, attrs_, messages) = d
                else:
                    (id_, x, y, domain, attrs_) = d[:5]

                # Get all (score, y') pairs:
                scores = [score(x, y_alt, self.phi, weights)+cost(y, y_alt) for y_alt in domain]
                # Get the maximal score:
                max_score = sorted(scores)[-1]
                error += max_score - score(x, y, self.phi, weights)
                # Compute the gradient of the objective function:
                grad = self.gradient(x_actual=x, x_alts=messages,
                                     y_actual=y, y_alts=domain,
                                     w=weights, verbose=verbose)
                # L2 regularization: subtract constant multiple of weight values
                if l2_coeff:
                    for f in set(weights.keys()):
                        grad[f] -= l2_coeff * weights[f]
                # Weight-update (a bit cumbersome because of the dict-based implementation):
                if self.use_adagrad:
                    for f in set(grad.keys()):
                        adagrad[f] += grad[f] ** 2
                        if adagrad[f] != 0.0:
                            dw = self.eta * grad[f] / np.sqrt(adagrad[f])
                            weights[f] += dw
                            update_mag += dw ** 2
                else:
                    for f in set(grad.keys()):
                        dw = self.eta * grad[f]
                        weights[f] += dw
                        update_mag += dw ** 2
            timing.end_task()
            if verbose: 
                print 'Error: %f' % error
                print 'Weight update magnitude: %f' % update_mag
            if error <= self.epsilon:
                if verbose:
                    print "Terminating after %s iterations; error is minimized." % iteration
                break
            if update_mag <= self.epsilon:
                if verbose:
                    print "Terminating after %s iterations; reached local minimum." % iteration
                break
        timing.end_task()
        return (weights, error, iteration, messages)


def log_softmax(a, axis=None):
    """Return the log of the softmax function applied to the scores given by `a`
    across the axis `axis` (default: softmax over all elements of `a`)."""
    return a - logsumexp(a, axis, keepdims=True)


class LRUCache(object):
    def __init__(self, compute_func, max_entries=None):
        self.compute_func = compute_func
        self.max_entries = max_entries

        self.lookup = OrderedDict()
    
    def __call__(self, *args):
        key = tuple(id(a) for a in args)
        if key in self.lookup:
            result = self.lookup[key]
            del self.lookup[key]
        else:
            result = self.compute_func(*args)
            if self.max_entries and len(self.lookup) >= self.max_entries:
                self.lookup.popitem(last=False)
        self.lookup[key] = result
        return result


def featurize_all(phi, y_alts, x_alts, weights):
    dims_map = {}
    names = []
    next_dim = 0

    vals_slices = []
    r_slices = []
    c_slices = []
    for y in y_alts:
        vals_slices.append([])
        r_slices.append([])
        c_slices.append([])
        for row, x in enumerate(x_alts):
            feat = phi(x, y)
            for name, val in feat.iteritems():
                vals_slices[-1].append(val)
                r_slices[-1].append(row)
                if name not in dims_map:
                    dims_map[name] = next_dim
                    names.append(name)
                    next_dim += 1
                c_slices[-1].append(dims_map[name])

    featurizations = [sparse.coo_matrix((vals, (r, c)),
                                        shape=(len(x_alts), next_dim))
                      for vals, r, c in zip(vals_slices, r_slices, c_slices)]

    weights_vec = sparse.coo_matrix(np.array([[weights[name] if name in weights else 0.0
                                               for name in names]]))
    return featurizations, weights_vec, names


def unvectorize(feat_vec, names):
    """Given features as a vector, return a dict pairing feature names
    with values."""
    return defaultdict(float, {name: feat_vec[i]
                               for i, name in enumerate(names)
                               if feat_vec[i] != 0})


class FeatureVectorizer(object):
    def __init__(self, phi, cache=True, messages=[], data=[], verbose=0):
        self.dims_map = {}
        self.names = []
        self.next_dim = 0
        self.phi = phi
        self.verbose = verbose

        if cache:
            self.mat_cache = LRUCache(self._compute_featurized)
        else:
            self.mat_cache = self._compute_featurized
        
        self.preallocate(messages, data)

    def preallocate(self, messages, data):
        """Pre-compute the dimensions for all possible features given a dataset `data`,
        and a set of possible `messages`."""
        if self.verbose >= 1:
            print 'FeatureVectorizer: preallocating with %d messages, %d examples' % \
                (len(messages), len(data))
        y_alts = {}
        for (_, _, _, domain, _) in data:
            for y_alt in domain:
                y_alts[str(y_alt)] = y_alt

        if self.verbose >= 1:
            print 'FeatureVectorizer: number of referents: %d' % (len(y_alts),)
        for i, x in enumerate(messages):
            if self.verbose >= 1 and i % 10 == 0:
                print 'FeatureVectorizer: preallocating message %d of %d' % (i, len(messages))
            for y in y_alts.values():
                for name in self.phi(x, y):
                    self.get_dim(name)

    def get_dim(self, name):
        if name not in self.dims_map:
            self.dims_map[name] = self.next_dim
            self.names.append(name)
            self.next_dim += 1    
        return self.dims_map[name]
    
    def get_name(self, dim):
        return self.names[dim]
    
    def num_dims(self):
        return self.next_dim

    def _compute_featurized(self, y, x):
        return self.vectorize(self.phi(x, y))
        
    def featurize(self, y, x):
        """Return a sparse 1xn matrix of feature values for the utterance `x`
        and the referent `y`."""
        return self.mat_cache(y, x)

    def featurize_all(self, y_alts, x_alts):
        """Given an iterable of referents `y_alts` and an iterable of utterances
        `x_alts`, return a list of scipy sparse matrices representing the feature
        values of (x, y), where each matrix in the list is for a different referent
        `y` in `y_alts` and each row in that matrix is a different utterance in
        `x_alts`."""
        return [sparse.vstack([self.featurize(y, x) for x in x_alts])
                for y in y_alts]

    def vectorize(self, feats):
        if isinstance(feats, dict):
            feats = [feats]
        triples = [(val, row, self.get_dim(name))
                   for row, feat in enumerate(feats)
                   for name, val in feat.iteritems()]
        if not triples:
            vals, r, c = [0], [0], [0]
        else:
            vals, r, c = zip(*triples)
        return sparse.coo_matrix((vals, (r, c)),
                                 shape=(len(feats), self.num_dims()))
        
    def unvectorize(self, feat_vec):
        """Given features as a vector, return a dict pairing feature names
        with values."""
        return defaultdict(float, {name: feat_vec[i]
                                   for i, name in enumerate(self.names)
                                   if feat_vec[i] != 0})

    def print_features(self, feat_vec, include_zeros=False):
        print(type(feat_vec))
        if hasattr(feat_vec, 'toarray'):
            feat_vec = np.squeeze(feat_vec.toarray())
        print(feat_vec.shape)
        feats = self.unvectorize(feat_vec)
        for k, v in feats.iteritems():
            if include_zeros or v != 0.0:
                print '%10s %s' % (v, k)
