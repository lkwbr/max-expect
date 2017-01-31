# Luke Weber, 11398889
# CptS 570, HW #5
# Created 11/21/2016

# Problem 1

import math
import operator
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import sys

class GaussianMixtureModel:
    """
    
    """

    # Data list, model list, and number of models, k
    data = []   # [0, j]
    models = [] # [0, i]
    k = 0

    def run_e(self):
        """
        Run expectation step
        """

        # For every model, compute p(x_i|y=mod)
        for mod in self.models:
            for idx, dat in enumerate(self.data):
                pr = mod.prob_point(dat, idx)
        
        # Do p(y=mod|x_j) for all mod:=[0, k] classes, and give
        # each cluster its own children (i.e. data points for
        # which the probability they belong to that class >= 0.5)
        # for current iteration of EM
        for mod in self.models:
            mod.children = []
            for i in range(len(self.data)):
                prob = self.prob_model(mod, i)
                mod.children.append(i)
            
    def run_m(self):
        """
        Run maximization step
        """

        # Diagnostics: make sure there are no alike children
        # between the two models
        intersect = list(set(self.models[0].children) &
                         set(self.models[1].children))        
        #assert intersect == []

        for mod in self.models:

            mean_nume = 0
            mean_denom = 0
            variance_nume = 0
            variance_denom = 0

            # Compute new mean
            for i in mod.children:
                p = self.prob_model(mod, i)
                mean_nume += p * self.data[i]
                mean_denom += p
            mod.mean = mean_nume / mean_denom

            # Compute new variance (using new mean)
            for i in mod.children:
                p = self.prob_model(mod, i)
                variance_nume += p * math.pow(self.data[i] - mod.mean, 2)
                variance_denom += p
            mod.variance = variance_nume / variance_denom

    def prob_model(self, model, i):
        """
        Get probability that given distribution contains the given point,
        i.e. p(model|point)
        """

        # NOTE: Let's say p(k) = 1 / k
        nume = model.probs[i] * (1 / self.k)
        denom = 0
        for m in self.models: denom += m.probs[i] * (1 / self.k)

        return nume / denom

    def get_model_range(self, model):
        """
        Return range of children
        """

        range_max = 0
        range_min = sys.maxsize

        for c in model.children:
            val = self.data[c]
            if val > range_max: range_max = val
            if val < range_min: range_min = val

        return [range_min, range_max]

    def __str__(self):
        """
        Returns string representation of this class, like information about
        all the clusters/models
        """
        
        this_str = ""
        
        for idx, mod in enumerate(self.models):
            this_str += "\tCluster #" + str(idx) + "\n"
            this_str += "\t\tMean: " + str(mod.mean) + "\n"
            this_str += "\t\tVariance: " + str(mod.variance) + "\n"
            this_str += "\t\tChildren: " + str(len(mod.children)) + "\n"
            this_str += "\t\tRange of children: " + \
                        str(self.get_model_range(mod)) + "\n"
            
        return this_str

    def __init__(self, data, k):
        """
        Constructor: store data points and generate k Gaussian models
        with random starting parameters
        """
        
        # Store member variables
        self.data = data
        self.k = k
        
        # Get range of data set (for random params)
        data_min = min(self.data)
        data_max = max(self.data)
        data_len = len(self.data)
    
        # Create k new Gaussian models
        for i in range(k):
            rand_mean = random.uniform(data_min, data_max)
            rand_variance = random.uniform(data_min, data_max) / 10
            gauss_model = GaussianModel(rand_mean, rand_variance,
                                        data_len, i)
            self.models.append(gauss_model)

class GaussianModel:
    """
    Info: https://en.wikipedia.org/wiki/Gaussian_function
    """

    # Model ID
    ident = -1

    # Standard Gaussian params
    mean = 0
    variance = 0
    
    # Probabilites of each data point, each corresponding to the ith
    # index of the original data list -- all initially 0
    probs = []

    # Stores indices of data points which currently fall under this
    # Gaussian distribution over all others
    children = []
    
    def prob_point(self, point, index):
        """
        Get probability that given point is contained in this Gaussian
        distribution! (i.e. p(point|b))
        """

        # Computing probability density function of a normally distributed
        # random variable 'point'
        # Info: https://en.wikipedia.org/wiki/Gaussian_function
        # NOTE: Using explicit line continuation with "\"
        probability = 1 / math.sqrt(2*math.pi*math.pow(self.variance, 2)) \
                      * math.exp(-1* \
                          math.pow(point - self.mean, 2) \
                          / (2*math.pow(self.variance, 2)))

        # Store that probability for later
        self.probs[index] = probability

        return probability
        
    def __init__(self, mean, variance, data_len, ident):
        self.mean = mean
        self.variance = variance
        self.probs = [0] * data_len
        self.ident = ident

def run_em(model):
    """
    Run Expectation Maximization algorithm with GMM
    """

    # TODO: Check for convergence

    print("Starting configuration:")
    print(model)

    print("Running EM iterations...")
    num_iter = 5

    x_list = model.data
    y_list = [0.1] * len(x_list)
    plt.plot(x_list, y_list)

    for mod in model.models:
        mu = mod.mean
        variance = mod.variance
        sigma = math.sqrt(variance)
        x = np.linspace(min(x_list), max(x_list), 100)
        plt.plot([mu], [0.4])
        plt.plot(x, mlab.normpdf(x, mu, sigma))
    
    for it in range(num_iter):

        print("\t[%d] --------------------------------------------------"
              % it)
        
        # Expectation step: compute probability that given point lies
        # in any one of our k classes
        model.run_e()

        # Maximization step: estimate the probabilities given the probability
        # that each point belongs to each class
        model.run_m()

        # Display via plots
        for mod in model.models:
            mu = mod.mean
            variance = mod.variance
            sigma = math.sqrt(variance)
            x = np.linspace(min(x_list), max(x_list), 100)
            plt.plot([mu], [0.4])
            plt.plot(x, mlab.normpdf(x, mu, sigma))
        #if it == 0: plt.show()
        #plt.show()

        # Show model information
        print(model)

    plt.show()

    for mod in model.models:
        mu = mod.mean
        variance = mod.variance
        sigma = math.sqrt(variance)
        x = np.linspace(min(x_list), max(x_list), 100)
        plt.plot([mu], [0.4])
        plt.plot(x, mlab.normpdf(x, mu, sigma))

    plt.show()

def preprocess(data_loc):
    """
    Converts original file data into list of floating point numbers
    """

    print("Preprocesing data...")
    
    # Get data into memory as list of lines
    with open(data_loc, 'r') as file:
        data_raw = file.read().splitlines()
        
    # Convert string float values to actual floats
    data = list(map(float, data_raw))
    
    return data

def main():
    """
    Essential process for running Expectation Maximization (EM) algorithm
    with 1-D Gaussian Mixture Models (GMMs):
        1. Place Gaussians randomly (params - mean and variance)
        2. For each data point, compute probability it belongs to each class
        3. Adjust parameters for each Gaussian to fit the points assigned to them
        4. If not converging, repeat step 1
    """

    # Preprocess
    data = preprocess("data/em_data.txt")

    # Construct initial model (with number of clusters k)
    k = 5
    model = GaussianMixtureModel(data, k)

    # Run EM algorithm until clusters converge,
    # noting the time it takes to do so
    time_start = time.clock()
    run_em(model)
    time_end = time.clock()
    print("Took %.2f seconds for clusters to converge"
          % ((time_end - time_start)))

# Kickoff!
main()
