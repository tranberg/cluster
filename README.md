This script calculates the clustering coefficient and degree distributions for a
particular kind of random network growth.

It is a work in progress. Not quite finished yet.

The growth process
------------------
Given an initial network a link is picked at random and a new node is attached to the two
nodes connected by the picked link.

After building a network the clustering coefficient and average node degrees are
calculated for later analysis.

Network sizes range from 10 to 1e6 nodes and can easily be changed. Because of the large
network size it requires about 12 GB of ram and about a minute on a 3.4 GHz CPU per
realisation of a 1e6 node network.

Files
-----
- cluster.py: Building of networks and various calculations.
- plotter.py: Uses output from cluster.py to produce figures.

Samples
-------
If the file plotter.py is run with the files in sample/ as input it will
produce nice figures for a run of 100 realisations.
