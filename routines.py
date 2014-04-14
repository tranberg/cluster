from __future__ import division
import numpy as np
from random import randint

# Contains useful functions for other scripts.

def where(n):
    """
    Return a list of ones. Used to populate a sparse matrix.
    """
    return np.ones(len(n))

def pick(a):
    """
    Pick a random index in the given list. Used to pick a random link in the
    network.
    """
    return randint(0,len(a)-1)

def add(row,col,n):
    """
    Adding n nodes to the network and updating the sparse adjacency matrix.
    """
    if n==0:
        return row,col
    for i in range(n):
        l = (len(col)-3)/2+3
        index = pick(row)
        r = row[index]
        c = col[index]
        col.append(l)
        col.append(l)
        row.append(r)
        row.append(c)
    return row,col

def ki(i,M):
    """
    Calculate the node degree of a single node from the adjacency matrix in
    sparse format
    """
    return M.sum(0)[0,i]
