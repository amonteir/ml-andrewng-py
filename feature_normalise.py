"""
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
"""

import numpy as np


def feature_normalise(x):
    x_normalised = x
    num_features = x.shape[1]
    mu = np.zeros((1, num_features))
    sigma = np.zeros((1, num_features))

    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: First, for each feature dimension, compute the mean
    %               of the feature and subtract it from the dataset,
    %               storing the mean value in mu. Next, compute the 
    %               standard deviation of each feature and divide
    %               each feature by it's standard deviation, storing
    %               the standard deviation in sigma. 
    %
    %               Note that X is a matrix where each column is a 
    %               feature and each row is an example. You need 
    %               to perform the normalization separately for 
    %               each feature. 
    %
    % Hint: You might find the 'mean' and 'std' functions useful.
    %  
    """
    print("{}".format(x.x0))

    for f in range(num_features):
        print("col={} , mean={} , sigma={}".format(f, np.mean(x.iloc[:, f]), np.std(x.iloc[:, f])))
        mu[0, f] = np.mean(x.iloc[:, f])
        sigma[0, f] = np.std(x.iloc[:, f])

        x_normalised.iloc[:, f] = x.iloc[:, f] - mu[0, f]
        x_normalised.iloc[:, f] = x_normalised.iloc[:, f] / sigma[0, f]

    return x_normalised, mu, sigma
