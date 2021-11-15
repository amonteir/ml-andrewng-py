"""
Compute cost for linear regression
%   J = compute_cost(x, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in x and y

"""


def compute_cost(x, y, theta, m):
    predictions = x.dot(theta)  # mx2 2x1 => mx1
    sqrErrors = (predictions - y) ** 2

    J = (1 / (2 * m)) * sum(sqrErrors)

    return J


def compute_cost_multi(x, y, theta, m):
    J = (1 / (2 * m)) * (x.dot(theta) - y).T.dot(x.dot(theta) - y)

    return J[0]
