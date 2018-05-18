#/usr/bin/env python

import numpy as np
from scipy import stats
from itertools import combinations
import bootstrap

def rownorm(mat):
    """Row normalization of a matrix"""
    return np.divide(mat.T, np.sum(mat, axis=1)).T
    
def colnorm(mat):
    """Column normalization of a matrix"""    
    return np.divide(mat, np.sum(mat, axis=0))

def safelog(vals):
    with np.errstate(divide='ignore'):
        return np.log(vals)

def confidence_interval(vals):
    if len(set(vals)) == 1:
        return (vals[0], vals[0])
    # In case bootstrap.py is missing or not working:
    # loc = np.mean(vals)
    # scale = np.std(vals) / np.sqrt(len(vals))
    # return stats.t.interval(0.95, len(vals)-1, loc=loc, scale=scale)        
    return bootstrap.ci(vals, method='bca')
    
def display_matrix(mat, rnames=None, cnames=None, title='', digits=4, latex=False):
    """Utility function for displaying strategies to standard output."""
    rowlabelwidth = 2 + max([len(x) for x in rnames] + [digits+2])
    cwidth = 2 + max([len(x) for x in cnames] + [digits+2])
    s = ""
    divider = ""
    linebreak = "\n"
    cmt = ""        
    if latex:
        divider = " & "
        linebreak = "\\\\\n"
        cmt = "% "
    # Divider bar of the appropriate width:
    s += cmt + "-" * ((cwidth * len(cnames)) + rowlabelwidth) + "\n"
    s += cmt + title + "\n"
    # Real table:
    if latex:
        s += "\\begin{tabular}[c]{ *{%s}{r} }\n" % (len(cnames)+1)
        s += r"\toprule" + "\n"
    mat = np.round(mat, digits)      
    # Matrix with even-width columns wide enough for the data:
    s += ''.rjust(rowlabelwidth) + divider + divider.join([str(s).rjust(cwidth) for s in cnames]) + linebreak
    if latex:
        s += r"\midrule" + "\n" 
    for i in range(mat.shape[0]):  
        s += str(rnames[i]).rjust(rowlabelwidth) + divider + divider.join(str(x).rjust(cwidth) for x in mat[i, :]) + linebreak
    if latex:
        s += r"\bottomrule" + "\n"
        s += r"\end{tabular}"
    print s
        
def powerset(x, minsize=0, maxsize=None):
    result = []
    if maxsize == None: maxsize = len(x)
    for i in range(minsize, maxsize+1):
        for val in combinations(x, i):
            result.append(list(val))
    return result

def mse(x, y):
    """Mean squared error"""
    #err = np.sqrt(np.sum((x-y)**2)/len(x))
    err = np.mean((x-y)**2)
    return err
