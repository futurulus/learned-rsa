import numpy as np
from matplotlib import pyplot as plot


def npenumerate(arr):
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        yield (it.multi_index, it[0])
        it.iternext()


def plot_matrix(mat, xlabels=None, ylabels=None, prefix=()):
    if len(mat.shape) < 2:
        plot_matrix(mat[np.newaxis, :], xlabels, ylabels, prefix)
    elif len(mat.shape) == 2:
        if prefix:
            print (prefix)
        im = plot.matshow(mat, fignum=False,
                          interpolation='none', cmap=plot.get_cmap('gnuplot'))
        if xlabels:
            plot.yticks(range(len(ylabels)), ylabels)
        if ylabels:
            plot.xticks(range(len(xlabels)), xlabels)
        for label in im.axes.xaxis.get_ticklabels():
            label.set_rotation(90)
        plot.show()
    else:
        for j in xrange(mat.shape[0]):
            plot_matrix(mat[j], xlabels, ylabels, prefix + (j,))


def print_matrix(mat):
    if isinstance(mat, list):
        print('list[%s]' % type(mat[0]))
        mat = np.array([m.toarray() for m in mat])
    elif hasattr(mat, 'toarray'):
        print(type(mat))
        mat = mat.toarray()
    else:
        print(type(mat))
    print(mat.shape)
    plot_matrix(mat)
    for i, v in npenumerate(mat):
        print '%s %s' % (i, v)
