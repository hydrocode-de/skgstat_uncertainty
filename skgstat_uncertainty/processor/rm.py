"""
Risk minimization
-----------------
Risk minimization is a inductive principle from machine learning.
It balances the model complexity and the fitting performance.
Risk refers to the the unkown performance of a trained model on a test set.
Model complexity is expressed as capacity, usually expressed as the VC-dimension.
This does only apply to classification problems and thus in this context
the variogram has to be expressed in terms of classification, or another
metric has to be utilized to express the model complexity.

This module implements the empirical risk as a metric, that the user is 
seeking to minimize. The empirical risk does take different loss functions, but is
generally similar to the RMSE metric as used here.

The second approach is structural risk minimization, which balances the empirical risk
of the model with its complexity.

We made several adaptions to the principles of risk minimization to use it in
the context of uncertainty-driven variogram fitting:

The MAE is defining the squared error in terms of 
"""
from typing import List, Tuple
import numpy as np
from skgstat import Variogram

from skgstat_uncertainty.models import VarioConfInterval
from skgstat_uncertainty.processor.dic import effective_parameters


def loss_d(y: float, interval: Tuple[float, float]) -> float:
    """"""
    if y < interval[0]:
        return interval[0] - y
    elif y > interval[1]:
        return y - interval[1]
    else:
        return 0


def loss_zero(y: float, interval: Tuple[float, float]) -> int:
    """"""
    return y > interval[0] and y < interval[1]


def empirical_risk(vario: Variogram, conf_interval: VarioConfInterval, method: str = 'mae') -> float:
    """
    Calculate the empirical risk of the fit using a loss function for 
    the given predictions and ground truth.
    There are different methods implemented:
      
      - MAE (mean absolute error)
      - binary
    
    Both return 0 if the variogram is entirely within the confidence interval.
    If any of the bins is outside of the confidence interval, the loss of 'binary' is
    the percent of bins outside. The mse method returns the mean absoulte deviation
    from the confidence interval.

    """
    # get the interval bounds
    bounds = np.array(conf_interval.spec['interval'])
    
    # get the models prediction
    #y_pred = vario.experimental
    y_pred = vario.fitted_model(vario.bins)

    # get the loss function
    loss = loss_zero if method.lower() == 'binary' else loss_d

    # map loss function
    e = np.fromiter(map(loss, y_pred, bounds), dtype=float)

    # check what to do with the errors
    if method.lower() == 'binary':
        return np.nanmean(e)
    elif method.lower() == 'mae':
        return np.nanmean(e)
    else:
        raise AttributeError(f"Unknown method: {method}")


def structural_risk(vario: Variogram, conf_interval: VarioConfInterval, others: List[Variogram], weight: float = 1, method: str = 'mae', pD_method='mean') -> float:
    """
    """
    # calculate the empirical risk
    er = empirical_risk(vario, conf_interval, method=method)

    # estimate the model complexity using the effective parameters
    pD = effective_parameters(others, method=pD_method)
    
    return er + weight * pD
