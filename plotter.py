from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import optimize

# Functions for fitting the degree distribution
def fitfunc(p,x):
    return p[0] + p[1] * x ** (p[2])
def errfunc(p,x,y):
    return (y - fitfunc(p, x))

# Loading some data
data = np.loadtxt('such_result_100_realisations')
k = np.load('node_degrees.npz')['k']
mins = data[0]
maxs = data[1]
means = data[2]

# Plotting of error bars
x = [10,100,1000,10000,100000,1000000]
fig,ax = plt.subplots(1,1,1)
ax.errorbar(x,means,[mins,maxs],xerr=None,marker='s',fmt='.')
ax.set_xlim(7, 2000000)
ax.set_ylim(.65,.775)
ax.set_xscale('log')
plt.xlabel('Nodes added to the original network')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient versus network growth')
ax.annotate(r'C = '+str("%.5f"% means[-1])+' ('+str("%.5f"% mins[-1])+', '
+str("%.5f"% maxs[-1])+')', xy=(1e6,.739), xytext=(1e3, .76),
arrowprops=dict(facecolor='black', shrink=0.1,width=2))

# Calculation of degree distributions
top = np.max(k)
H = np.zeros((len(k),top))
bins = np.linspace(0,top+1,top+2)
for i in range(len(k)):
    H[i],bins = np.histogram(k[i],bins)
k = np.mean(H,0)    # Averaging over the histograms for all realisations

# Polynomial fitting to the averaged degree distribution

# Below here needs fixing!
xdata = bins
ydata = k

def func(a,x,b):
    return a*x**b

opt,cov = optimize.curve_fit(func,xdata,ydata)



# plot degree distribution on a log-log scale

plt.show()
#plt.savefig('wow.png')
