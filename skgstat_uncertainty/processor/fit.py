"""
Various metrics to assess the quality of a variogram model parameterization.
These metrics are key to manually parameterizing variogram models in the context
of uncertainty bounds.
The metrics can be used solely or jointly to filter a number of parameterizations,
to select the parameter candidates best represeting the uncertain variogram with
respect to uncertainty bounds.

The implemented metrics are:

* RMSE: the root mean squared error adjusted to the confidence interval.
* cv: leave-one-out cross-validation by using the parameters to predict an observation based on the sample using kriging
* dic: DIC of a collection of parameterization associated to a specific model.
* aic / bic: AIC / BIC of a collection of parameterizations associated to a specific model
* empirical / structural risk of a parameterization with respect to all other parameter sets calculated for the same model type.

.. warning::
    We are not sure if the AIC / BIC calculation is scientifically valid if the optimal parameter set
    may possibly not be included in the collection. Strictly speaking it applies only to the optimal 
    parameter set.

.. note::
    The metrics dic / aic / bic are metrics on model type level.

.. note:: 
    The empirical / structural risk metrics are always calculated with respect to the whole
    collection of parameters for the given model type. With each new parameterization in the
    database (for the current confidence interval instance) the values for all other parameterizations
    need to be updated.

"""
from typing import List, Union
import numpy as np
from numpy.random import choice
from skgstat import models, Variogram
from gstools import Krige

from skgstat_uncertainty.models import VarioParams, VarioConfInterval, VarioModel
from skgstat_uncertainty.utils.vario_tools import rebuild_variogram
from skgstat_uncertainty.processor.dic import DIC, deviance as dic_deviance
from skgstat_uncertainty.processor import rm


def rmse(vario: VarioParams, interval: VarioConfInterval, params: dict) -> float:
    """
    Adjusted RMSE for the params with respect to the confidence interval of 
    the passed empirical variogram. The RMSE is adjusted to yield 0 for bins
    within the uncertainty bounds, even if there is a deviance to the 
    experimental semi-variance values.

    Parameters
    ----------
    vario : VarioParams
        Empirical variogram estimation from the database.
    interval : VarioConfInterval
        Uncertainty bounds representation from the database.
    params : dict
        The variogram model parameterization to be assessed
    
    Returns
    -------
    rmse : float
        RMSE value for params

    """
    # build the variogram instance
    variogram = vario.variogram

    # ge the model function and bins
    model = params['model']
    model_func = getattr(models, model)
    bins = variogram.bins

    # get the interval itself
    bounds = interval.spec['interval']

    # build
    if params.get('shape') is None:
        func = lambda h: model_func(h, params.get('range'), params.get('sill'), params.get('nugget'))
    else:
        func = lambda h: model_func(h, params.get('range'), params.get('sill'), params.get('shape'), params.get('nugget'))

    # apply the current model parameters
    y = [func(h) for h in bins]

    # get the difference to each bin
    se = []
    for _y, bnd in zip(y, bounds):
        if _y > bnd[0] and _y < bnd[1]:
            se.append(0)
        else:
            se.append(np.power(np.nanmin([np.abs(_y - bnd[0]), np.abs(_y - bnd[1])]), 2))
    
    # return
    return np.sqrt(np.nanmean(se))


def cv(vario: Union[VarioParams, Variogram], params: dict, k: int = 1, max_iter: int = None) -> float:
    """
    Leave-one-out cross-validation for the params for an estimated
    empirical variogram. Each observation is interpolated using
    ordinary kriging using all other observations. The RMSE of
    all interpolations is returned.

    Parameters
    ----------
    vario : VarioParams
        Empirical variogram estimation from the database.
    params : dict
        The variogram model parameterization to be assessed
    k : 1
        Number of points to be omitted in each iteration.
        Defaults to 1, if you change this number, it is not
        a leave-one-out cv anymore.
    max_iter : int
        Maximum number of iterations. If None (default) the
        length of observations is used. If smaller, the result
        is not deterministic anymore.
    
    Returns
    -------
    rmse : float
        RMSE value for all interpolated observations.

    """

    # build the variogram instance
    variogram = rebuild_variogram(vario, params)

    # get the gstools covmodel instance
    covmodel = variogram.to_gstools()

    # set up the data arrays
    coords = variogram.coordinates
    values = variogram.values
    n = len(values)

    if max_iter is None:
        max_iter = n
    
    # determine the number of sets to calculate
    n_sets = int(k * max_iter)
    if n_sets > n:
        n_sets = n


    # create a shuffle index for all positions - but limit to n_sets
    idx = choice(range(n), size=n, replace=False)

    # append the mean error on each round
    err = []

    # go for the sets
    for iset in range(0, n_sets, k):
        # get the shuffled data
        set_idx = idx[iset : iset + k]
        x = np.array([c for i, c in enumerate(coords) if i not in set_idx]).T
        y = [v for i, v in enumerate(values) if i not in set_idx]
        
        # build the kriging model
        krige = Krige(covmodel, x, y, fit_variogram=False)
        y_pred = krige(coords[set_idx].T)

        # calculate the error
        e = np.mean(np.abs(y_pred - values[set_idx]))
        err.append(e)

    return np.mean(err) 


def aic(vario: Union[VarioParams, Variogram], params: dict) -> float:
    # get the fitted parameters variogram instance
    variogram = rebuild_variogram(vario, params)

    return variogram.aic


def bic(vario: Union[VarioParams, Variogram], params: dict) -> float:
    # get the fitted parameters variogram instance
    variogram = rebuild_variogram(vario, params)

    return variogram.bic


def dic(vario: Union[VarioParams,Variogram], params: dict, varioparams: List[VarioModel], method='mean') -> float:
    # build the current variogram instance
    variogram = rebuild_variogram(vario, params)

    # turn all VarioParams into Variograms
    variograms = [variogram, *[v.variogram for v in varioparams]]

    # claculate the DIC
    return DIC(variograms, method)


def idic(vario: Union[VarioParams, Variogram], params: dict, varioparams: List[VarioModel], method='mean', pop_dic: float = None) -> float:
    # calcualte the DIC of the current model
    if pop_dic is None:
        pop_dic = dic(vario, params, varioparams, method)

    # filter the list of params
    filtered_varios = [v.variogram for v in varioparams]

    # claculate the DIC for all except the requested model
    filtered_dic = DIC(filtered_varios, method)

    # return
    return pop_dic / filtered_dic


def deviance(vario: Union[VarioParams, Variogram], params: dict) -> float:
    # build the variogram instance
    variogram = rebuild_variogram(vario, params)

    # return
    return dic_deviance(variogram)


def empirical_risk(vario: Union[VarioParams, Variogram], interval: VarioConfInterval, params: dict, method: str = 'mae') -> float:
    # build the variogram instance
    variogram = rebuild_variogram(vario, params)

    return rm.empirical_risk(variogram, conf_interval=interval, method=method)


def structural_risk(vario: Union[VarioParams, Variogram], interval: VarioConfInterval, params: dict, models: List[VarioModel], weight: float = 1., method: str = 'mae', pD_method: str = 'mean') -> float:
    """"""
    # build the variogram instance
    variogram = rebuild_variogram(vario, params)

    # build the other variograms
    others = [m.variogram for m in models]

    return rm.structural_risk(variogram, interval, others, weight, method, pD_method)
