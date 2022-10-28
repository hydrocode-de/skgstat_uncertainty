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

The MAE is defining the squared error in terms of uncertainty bounds. As soon as
the empirical semi-variance associated to a bin falls into the uncertainty bound,
the MAE will be zero. That means, the MAE can be 0, even if there is a deviation 
from the prediction and experimental semi-variance value.

Additionally, we define a *binary* loss function for empirical risk. This will
return a 0 (False) if the prediction is outside of the uncertainty bounds and
1 (True) elsewise. The metric aggregated by empirical risk is then the mean
value of the binary loss, which is a number between 0 and 1. This can be interpreted
as the  percentage of bins predicted outside the uncertainty bounds.

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
    Structural risk minimization adapted after original publication by Vapnik et al. (1974).
    The structural risk minimization is a metric that balances the empirical risk of all passed
    parameterization and the complexitiy of the associated variogram model into a combined 
    metric.
    One can interpret the empirical risk part as the fitting error. Model complexity is
    assessed by the *effective parameters* as used in the DIC metric (Spiegelhalter et al. 2002)

    Parameters
    ----------
    variogram : skgstat.Variogram
        Main variogram instance to be assessed. The empirical risk is calculated
        for this instance.
    conf_interval : VarioConfInterval
        Uncertainty bounds instance
    others : list
        List of :class:`Variograms <skgstat.Variogram>` representing all parameterizations for
        the current model.
    weight : float
        Factor weighting empirical risk and model complexity
    method : str
        Can be either 'mae' or 'binary' for calculating the empirical risk

    See Also
    --------
    empirical_risk

    Returns
    -------
    srm : float
        The calcualted structural risk of the parameters, in the context of 
        all parameterizations of the same model.

    """
    # calculate the empirical risk
    er = empirical_risk(vario, conf_interval, method=method)

    # estimate the model complexity using the effective parameters
    pD = effective_parameters(others, method=pD_method)
    
    return er + weight * pD
