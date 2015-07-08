import numpy as np
import matplotlib.pyplot as plt

C = np.load('TTC_1000.npy')
runs = 1000

x = []
for N in [10, 100, 1000, 10000, 100000]:
    if N < 100000:
        top = N * 10
    else:
        top = N * 10 + 3 * N
    for n in range(N, top, N):
        x.append(n)

means = np.mean(C, 1)
mins = np.min(C, 1)
maxs = np.max(C, 1)
stds = np.std(C, 1)

print 'Mean: ', "%.5f" % means[-1]
print 'Plus: ', "%.5f" % (maxs[-1] - means[-1])
print 'Minus: ', "%.5f" % (means[-1] - mins[-1])

meanplot, = plt.plot(x, means, '-', lw=1.5, color="#000099")
minplot, = plt.plot(x, mins, '-k', lw=1, alpha=0.2)
plt.plot(x, maxs, '-k', lw=1.2, alpha=0.2)
stdplot, = plt.plot(x, means + stds, ':k', lw=1.5, alpha=.6)
plt.plot(x, means - stds, ':k', lw=1.5, alpha=.6)

for r in range(48):
    plt.plot([x[r] for i in range(runs)], C[r], '.k', alpha=.02)
scatter, = plt.plot([-10 for i in range(runs)], C[r], '.k')

plt.legend([meanplot, minplot, stdplot, scatter],
           ['Mean C', 'Max/min C', r'$\sigma$', 'Simulations'], loc=1)

plt.xscale('log')
plt.axis([9, 1.1 * x[-1], min(mins), max(maxs)])
plt.xlabel('Number of nodes in network')
plt.ylabel('Clustering coefficient, C')
plt.savefig('scatter_' + str(runs) + '.png', bbox_inches='tight')
