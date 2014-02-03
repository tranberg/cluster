from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import optimize

# Loading some data
data = np.loadtxt('such_result_100_realisations')
k = np.load('avk.npz')['k']
mins = data[0]
maxs = data[1]
means = data[2]

# Plotting of error bars
x = [10,100,1000,10000,100000,1000000]
fig,ax = plt.subplots()
ax.errorbar(x,means,[mins,maxs],xerr=None,marker='s',fmt='.')
ax.set_xlim(7, 2000000)
ax.set_ylim(.99*(means[0]-mins[0]),1.01*(means[0]+maxs[0]))
ax.set_xscale('log')
plt.xlabel('Nodes added to the original network')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient versus network growth')
ax.annotate(r'C = '+str("%.5f"% means[-1])+' ('+str("%.5f"% mins[-1])+', '
+str("%.5f"% maxs[-1])+')', xy=(1e6,.739), xytext=(1e3, .76),
arrowprops=dict(facecolor='black', shrink=0.1,width=2))

# Polynomial fitting to the averaged degree distribution
top = len(k)
bins = np.linspace(1,top,top)

xdata = bins
ydata = k

plt.figure()
def func(x,a,b):
    return a*(x**b)

opt,cov = optimize.curve_fit(func,xdata[20:500],ydata[20:500],p0=[1e7,-3])
a=opt[0]
b=opt[1]

def ys(x):
    return 1e7*x**-3

par = r'$\sim$'+r'$k^{'+str("%.3f"% b)+r'}$'
plt.loglog(xdata,ydata,'-')
plt.loglog(xdata[:1e3],ys(xdata)[:1e3])
plt.loglog(xdata[:1e3],func(xdata,a,b)[:1e3],'-')
plt.legend(('Degree distribution',r'$k^{-3}$','Fit: '+par),loc="best")
plt.title(r'Degree distribution of a network with $10^6$nodes')
plt.xlabel('Node degree')
plt.ylabel('Number of nodes')

plt.show()
