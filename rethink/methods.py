import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class GridApproximation():
    def __init__(self, prior, k, n, g):
        self.k = k
        self.n = n
        self.prior = prior
        self.g = g
        self.i = 10000

    
    def posterior(self):
        """
        Calculate posterior using grid approximation

        :param k: number of succeses
        :param n: number of trials
        :param prior: prior, same size as g
        :param g: size of the grid
        """ 
        # create a grid over which we will calculate the likelihood
        self.p_grid = np.linspace(0, 1, num = self.g)
        # calculate the probability of observing the data
        self.likelihood = stats.binom.pmf(self.k,self.n,p = self.p_grid)
        # multiply with prior
        unst_posterior = self.prior * self.likelihood
        # standardize
        self.stand_posterior = unst_posterior / np.sum(unst_posterior)
        
        #sample from posterior
        np.random.seed(42)
        self.samples = np.random.choice(a=self.p_grid,size=self.i,replace=True,p=self.stand_posterior)

        #calculate posterior predictive distribution
        self.posterior_predictive_dist = stats.binom.rvs(n=self.n,p=self.samples,size=self.i)

    def viz(self):
        """
        Vizualize posterior 
        """
        fig, ax = plt.subplots(1, 2, figsize = (10,4))
        ax[0].plot(self.p_grid, self.prior)
        ax[0].set_title("Prior")
        ax[1].plot(self.p_grid, self.stand_posterior)
        ax[1].set_title("Posterior: k={}, n={}".format(self.k,self.n))
        plt.show()


