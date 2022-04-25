from typing import List

import numpy as np
from scipy.stats import mode
from skgstat import Variogram
from skgstat.util.likelihood import get_likelihood

from skgstat_uncertainty.utils.vario_tools import parameterized_clone


def DIC(variograms: List[Variogram], method='mean') -> float:
    """
    Calculate the DIC (deviance information criterion) [201]_ for the given list of
    variograms. Each variogram instance is interpreted as a sample parameterization
    for the used model. You have to make sure, that each instance shares the same
    model and the parameter array has identical shape.
    The DIC supports different methods to calculate the effective parameters.
    The effective parameters can be calculated as the difference between the posterior
    mean deviance and the deviance of the model parameterized with inferred parameters.
    These parameters can be inferred by the median, mean or mode of the model. 
    An alternative approach is to use Gelman's extenstion to the DIC [202]_, which uses
    the deviance variance to estimate the effective parameters independent of the
    parameterization.

    Parameters
    ----------
    variograms : List[Variogram]
        List of variograms to calculate the DIC for.
    method : str
        Method to calculate the effective parameters. 
        Can be 'mean', 'median', 'mode' or 'gelman'.
        Default is 'mean'.

    References
    ----------
    [201]   Spiegelhalter, D. J., Best, N. G., Carlin, B. P., & Van Der Linde, A. (2002).
            Bayesian measures of model complexity and fit. Journal of the royal statistical
            society: Series b (statistical methodology), 64(4), 583-639.
    [202]   Gelman, Andrew; Carlin, John B.; Stern, Hal S.; Rubin, Donald B. (2004).
            Bayesian Data Analysis: Second Edition. Texts in Statistical Science.
            CRC Press. ISBN 978-1-58488-388-3.
    """
    # get the posterior mean deviance
    mean_deviance = posterior_mean_deviance(variograms)

    # get the number of effective parameters
    eff_params = effective_parameters(variograms, method)

    return mean_deviance + eff_params


def deviance(vario: Variogram, C: float = None) -> float:
    """
    Calculate the information deviance for the used parameters of the given
    model. This deviance is used for the deviance information criterion, which
    compares the likelihood of different models, given the effective parameters.
    If many models are compared (like with DIC), then C does not have to be
    specified, because it cancels out. If the deviance will be used directly,
    you need to specify C.
    """
    # get the likelihood function for this instance
    log_like = get_likelihood(vario)

    # format the params
    theta = vario.parameters

    # information deviance
    D = 2 * log_like(theta)

    if C is not None:
        return D + C
    else:
        return D


def posterior_mean_deviance(variograms: List[Variogram]) -> float:
    """Posterior mean deviance.
    
    """
    # get the deviances
    deviances = [deviance(vario) for vario in variograms]

    # calculate the posterior mean
    return np.nanmean(deviances)



def effective_parameters(variograms: List[Variogram], method='mean') -> float:
    
    """Effective parameters
    Calculate the effective parameters for the given list of models. Following
    Spiegelhalter (2002), this can be calculated by the posterior mean, median
    or mode of the model parameters or substituted by the extenstion of 
    Gelman et al. (2004), who suggest deriving it from the deviance variance.
    
    """
    if method.lower() == 'gelman' or method.lower() == 'var':
        return _effective_gelman(variograms)
    elif method.lower() == 'celeux':
        raise NotImplementedError
    else:
        return _effective_spiegelhalter(variograms, method)


def _effective_spiegelhalter(variograms: List[Variogram], method='mean') -> float:
    # we need the mean deviances first
    mean_deviance = posterior_mean_deviance(variograms)

    # extract the model parameters and stack into a parameter matrix
    Theta = np.stack([v.parameters for v in variograms], axis=1).T
    
    # get the the deviance of the parameters
    # this is estamited from the distribution of parameters
    if method.lower() == 'mean':
        theta = np.mean(Theta, axis=0)
    elif method.lower() == 'median':
        theta = np.median(Theta, axis=0)
    elif method.lower() == 'mode':
        theta = mode(Theta, axis=0).mode.flatten()
    else:
        raise AttributeError('The method is not known.')

    # build a new model using these parameters
    vario = parameterized_clone(variograms[0], theta)

    # the effective parameters are posterior mean deviance minus the deviance of
    # the model of aggregated parameters

    return mean_deviance - deviance(vario)


def _effective_gelman(variograms: List[Variogram]) -> float:
    # get all deviances
    deviances = [deviance(vario) for vario in variograms]

    return np.nanvar(deviances) / 2.


# add gelman et al 2004 and celeux et al 2006; DIC3 p. 655

# pD = mean(D(theta)) - D(theta)
# DIC = mean(D(theta)) + pD
