from __future__ import division
import sys
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse.linalg import eigsh

# This script calculates clustering coefficients for very large networks.
# With n # = 1e6 it uses about 11.5 GB RAM. On one core 3.4 GHz it takes about
# 1.5 hours to complete with 100 realisations and 1e6 nodes. It also saves
# average node degrees for each network averaged over all runs.

runs = 100   # Number of realizations

# Some useful functions
def where(n):
    return np.ones(len(n))

def pick(a):
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


# Here we build a network, calculate the clustering coefficients and the node
# degrees. This is repeated "runs" times for each of the network sizes specified
# in the outer loop.
mins = []
maxs = []
means = []
kk = []
for n in [10,100,1000,10000,100000,1000000]:
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

        CC = []
        j = int(2*(n+2)-1)
        nodes = 3+(j-3)/2
        tdata = np.concatenate((data[:j],data[:j]))
        trow = np.concatenate((row[:j],col[:j]))
        tcol = np.concatenate((col[:j],row[:j]))
        A = csr_matrix((tdata,([trow,tcol])))
        
        """
        This could be redone to save memory:
        In stead of building entire A, just build the upper triangle of A, call
        it a, then do c=a*a'*a. c.diagonal() is the equivalent to C.diagonal()
        further down.
        """

        print 'Done building A'
        """
        Here we should do the following:
        Keep A in memory.
        Calculate AA=A*A and keeping thins in memory
        We now calculate the diagonal of AAA=AA*A one element at a time and
        saving these elements in a file. This should save usage of RAM.
        We now have C
        When this works the code should be altered to support several CPU-cores
        """
    
        """
        Here is a failed attempt
        vals,P = eigsh(A,n-1)
        B = dia_matrix((vals,0),shape=(n-1,n-1),dtype=np.uint8)
        Atre = P*(B**3)*(P**-1)
        print Atre.diagonal()
        #print A[0].todense()
        """
        
        print 'Calculating clustering coefficient'
        
        """ new attempt """
        C = []
#        for q in range(n+3):
#            AA = np.zeros(n+3)
#            for w in range(n+3):
#                "do the square - one column at a time"
#                AA[w] = A[q].dot(A[w].T).todense()
#            "do the cube - one diagonal element at a time"
#            C.append((AA.dot(A[q].T.todense())).item(0))
#            print '\rProgress of cubing:',(q+1),'/',(n+3),
#            sys.stdout.flush()
        AAA = A*A
#        for ll in range(n+3):
#            C.append((AAA[ll]*A[ll].T).todense())
#            print '\rProgress of cubing:',(ll+1),'/',(n+3),
#            sys.stdout.flush()

        C = AAA*A
        C = C.diagonal()

        k = A.sum(0)    # node degrees
        if n == 1000000:
            kk.append(k)
            print "yes"
        f = np.multiply(k,(k-1))
        W = C/f
        WW = W.sum()/nodes

#        for i in range(int(nodes)):
#            C[i] = C[i]/(ki(i,A)*(ki(i,A)-1))
#            print '\rProgress of clustering coefficient:',(i+1),'/',(n+3),
#            sys.stdout.flush()
#        CC.append((1/nodes)*sum(C))
#        tc.append(CC)
        tc.append(WW)
#        print 'Done.'
    
    mins.append(np.min(tc))
    maxs.append(np.max(tc))
    means.append(np.mean(tc))

mins = np.abs(np.subtract(mins,means))
maxs = np.subtract(maxs,means)

# Save data to file for later plotting.
np.savetxt('such_result_'+str(runs)+'_realisations',[mins,maxs,means])
np.savez('node_degrees',k=kk)
