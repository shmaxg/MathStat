import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, dist_type, params):
        self.n = None
        self.dist_type = dist_type
        self.params = params
        self.x = []
        self.order_stats = []

    def generate(self, n):
        self.n = n
        if self.dist_type == 'uniform':
            self.x = rnd.uniform(self.params['a'], self.params['b'], (n,))
        elif self.dist_type == 'normal':
            self.x = rnd.normal(self.params['mu'], np.sqrt(self.params['sigma2']), (n,))
        elif self.dist_type == 'exponential':
            self.x = rnd.exponential(1.0/self.params['lam'], (n,))
        elif self.dist_type == 'bernoulli':
            self.x = rnd.binomial(1, self.params['p'], (n,))
        elif self.dist_type == 'binomial':
            self.x = rnd.binomial(self.params['N'], self.params['p'], (n,))
        else:
            raise NotImplementedError

    def order(self):
        self.order_stats = np.sort(self.x)

    def plot_hist(self):
        plt.hist(self.x, density=True)
        plt.show()


class EmpiricalDistribution:
    def __init__(self, x):
        self.x = np.array(x).copy()
        self.x.sort()
        self.n = len(self.x)
        self.F = [k / self.n for k in range(self.n)]
        # F[k] = F(x(k)-0) = F(x(k)) т.к. ф.р. непрер. слева

    def plot(self):
        plt.step(self.x, self.F, 'C0')
        plt.step([min(self.x)-1, min(self.x)], [0, self.F[0]], 'C0')
        plt.step([max(self.x), max(self.x)+1], [self.F[-1], 1], 'C0')
        plt.axis('equal')
        plt.show()

    def calc_stat_kolm_smir(self, F):
        n = len(self.x)
        dnp = max([abs((k+1)/n-F(self.x[k])) for k in range(n)])
        dnm = max([abs(k/n-F(self.x[k])) for k in range(n)])
        return max([dnp, dnm])