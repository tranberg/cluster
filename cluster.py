from __future__ import division
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dia_matrix
from routines import *  # see routines.py

# This script calculates clustering coefficients for very large networks.
# With n = 1e6 it uses about 11.5 GB RAM. On one core 3.4 GHz it takes about
# 1.5 hours to complete with 100 realisations and 1e6 nodes. It also saves
# average node degrees for the largest network averaged over all runs.

runs = 100   # Number of realizations

# Here we build a network, calculate the clustering coefficients and the node
# degrees. This is repeated "runs" times for each of the network sizes specified
# in the outer loop.
mins = []
maxs = []
means = []
kk = []
for n in [10,100,1000,10000,100000,1000000]: # network sizes
    tc = []
    print 'Building network with',n,'nodes.'
    for k in range(runs):
        print 'Run',k+1,'of',str(runs)
        
        # Initial network
        irow = [0,0,1]
        icol = [1,2,2]

        # Adding nodes
        row,col,data=[],[],[]
        row,col = add(irow,icol,n)
        data = where(row)

        # Building the complete adjacency matrix
        j = int(2*(n+2)-1)
        nodes = 3+(j-3)/2
        tdata = np.concatenate((data[:j],data[:j]))
        trow = np.concatenate((row[:j],col[:j]))
        tcol = np.concatenate((col[:j],row[:j]))
        A = csr_matrix((tdata,([trow,tcol])))
        
        print 'Done building A'
        print 'Calculating clustering coefficient'
        
        # Calculating node degrees and clustering coefficients.
        C = []
        AAA = A*A               # Matrix multiplication split in two in order
        C = AAA*A               # to save memory.
        C = C.diagonal()

        k = A.sum(0)            # Calculating node degrees.
        if n == 1000000:        # Saving node degrees for the largest network.
            kk.append(k)
            print "yes"
        f = np.multiply(k,(k-1))
        W = C/f
        WW = W.sum()/nodes      # Total clustering coefficient for the network.
        tc.append(WW)
    
    mins.append(np.min(tc))
    maxs.append(np.max(tc))
    means.append(np.mean(tc))

mins = np.abs(np.subtract(mins,means))
maxs = np.subtract(maxs,means)

# Averaging over all realisations in order to obtain an average degree
# distribution.
top = np.max(kk)
H = np.zeros((len(kk),top))
bins = np.linspace(0,top+1,top+1)
for i in range(len(kk)):
    H[i],bins = np.histogram(kk[i],bins)
k = np.mean(H,0)

# Save data to file for later plotting.
np.savetxt('such_result_'+str(runs)+'_realisations',[mins,maxs,means])
np.savez('avk',k=k)
