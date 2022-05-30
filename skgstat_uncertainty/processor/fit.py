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
