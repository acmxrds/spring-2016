from math import sqrt
import logging

from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('GBM')


def gbm(nodes, node2nghbs, brown_dist, seeds,
        T, delta, drift, volatility):
    """Geometric Brownian motion (GBM) propagation by
    Jin et al., KDD 2014."""
    newly_affected = list(seeds)
    infected = {node: -1 for node in seeds}
    infect_init = {}

    for node in infected:
        for nghb in node2nghbs[node]:
            if nghb in infected:
                continue
            infect_init[node, nghb] = 0
        newly_affected.remove(node)

    t = 0
    while t <= T:
        _log.info('Time: %d' % t)
        for node in infected.keys():

            if node in newly_affected:
                for nghb in node2nghbs[node]:
                    if nghb in infected:
                        continue
                    infect_init[node, nghb] = 0
                newly_affected.remove(node)

            for nghb in node2nghbs[node]:
                if nghb in infected:
                    continue
                infect_init[node, nghb] = infect_init[node, nghb] + delta
                tij = infect_init[node, nghb]
                trust = norm.rvs(loc=(drift - volatility*2/2.)*tij,
                                 scale=volatility*sqrt(tij))
                if trust >= brown_dist[node, nghb]:
                    newly_affected.append(nghb)
                    infected[nghb] = t

        t += delta

    return infected


nodes = [1,2,3,4,5,6,7,8]
node2nghbs = {1: [3,4,5], 2: [1,6], 3: [2], 4: [1,3,5], 5: [4],
              6: [2,7,8], 7: [], 8: [6,7]}
mention_freq = {(1,3): 2, (1,4): 1, (1,5): 4, (2,1): 5, (2,6): 2,
                (3,2): 2, (4,1): 6, (4,3): 8, (4,5): 3, (5,4): 3,
                (6,2): 1, (6,7): 4, (6,8): 1, (8,6): 3, (8,7): 1 }

gamma = 2.
def bdist(i, j, gamma):
    wij = mention_freq.get((i,j), 0) + 1.
    wji = (mention_freq.get((j,i), 0) + 1)**gamma
    d = 1./(wij * wji)
    return d
brown_dist = {(i,j) : bdist(i, j, gamma) for i, j in mention_freq.keys()}

seeds = [5]

T = 10
delta = 1
drift = 1.
volatility = 1.5

infected = gbm(nodes, node2nghbs, brown_dist,
               seeds, T, delta, drift, volatility)
print 'Infected: %s' % ', '.join(
        '%d: %4.2f' % item for item in infected.items())
