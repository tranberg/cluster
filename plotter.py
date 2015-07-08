import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def func(x, a, b):
    return a * (x ** b)


def ys(x):
    return 1e7 * x ** -3

# Loading some data
data = np.loadtxt('results_1000_realisations')
k = np.load('avk.npz')['k']
print k.shape
mins = data[0]
maxs = data[1]
means = data[2]

# Plotting of error bars
x = []
for N in [10, 100, 1000, 10000, 100000]:
    if N < 100000:
        top = N * 10
    else:
        top = N * 10 + 3 * N
    for n in range(N, top, N):
        x.append(n)

plt.figure(figsize=(13, 6))
ax = plt.subplot(121)
ax.errorbar(x, means, [mins, maxs], xerr=None, marker='s', fmt='.')
ax.set_xlim(7, 2000000)
ax.set_ylim(.99 * (means[0] - mins[0]), 1.01 * (means[0] + maxs[0]))
ax.set_xscale('log')
plt.xlabel('Nodes added to the original network')
plt.ylabel('Clustering coefficient')
plt.title('Clustering coefficient versus network growth')
ax.annotate(r'C = ' + str("%.5f" % means[-1]) + ' (' + str("%.5f" % mins[-1]) + ', '
            + str("%.5f" % maxs[-1]) + ')', xy=(1.2e6, .74), xytext=(1e3, .76),
            arrowprops=dict(facecolor='black', shrink=0.1, width=2))

# Polynomial fitting to the averaged degree distribution
top = len(k)
bins = np.linspace(1, top, top)

xdata = bins
ydata = k

opt, cov = optimize.curve_fit(func, xdata[20:500], ydata[20:500], p0=[1e7, -3])
a = opt[0]
b = opt[1]

ax = plt.subplot(122)
par = r'$\sim$' + r'$k^{' + str("%.3f" % b) + r'}$'
ax.loglog(xdata, ydata, '-')
ax.loglog(xdata[:1e3], ys(xdata)[:1e3])
ax.loglog(xdata[:1e3], func(xdata, a, b)[:1e3], '-')
plt.legend(('Degree distribution', r'$k^{-3}$', 'Fit: ' + par), loc="best", fontsize=11)
plt.title(r'Degree distribution of a network with $1.2 \cdot 10^6$nodes')
plt.xlabel('Node degree')
plt.ylabel('Number of nodes')
plt.axis([1, 1000, 1, 1e7])

plt.savefig('plots.png', bbox_inches='tight')
