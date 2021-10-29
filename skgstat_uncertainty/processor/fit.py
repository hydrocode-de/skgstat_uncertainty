from typing import Tuple, List
import numpy as np
from skgstat import models

from skgstat_uncertainty.models import VarioParams, VarioConfInterval


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


def cv(vario: VarioParams, params: dict) -> float:
    # build the variogram instance
    variogram = vario.variogram

    # set the model
    variogram.model = params['model']

    # apply the parameters
    variogram.fit(
        method='manual',
        range=params.get('range'),
        sill=params.get('sill'),
        nugget=params.get('nugget'),
        shape=params.get('shape')
    )

    # perform the cross-validation
    _cv = variogram.cross_validate()

    return _cv
