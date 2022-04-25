from typing import Union, List
from skgstat import Variogram

from skgstat_uncertainty.models import VarioParams



def rebuild_variogram(vario: Union[VarioParams, Variogram], params: dict) -> Variogram:
    """
    Use the given parameters dictionary and build a new variogram instance.
    The VarioParams instance, however, does not include the newly estimated 
    parameters for the theoretical variogram model. Thus, a `skgstat.Variogram`
    instance is created and the params are applied to it.

    """
    # build the variogram instance
    if isinstance(vario, VarioParams):
        variogram = vario.variogram
    else:
        variogram = vario.clone()

    # set to manual fitting
    variogram.fit_method = 'manual'

    # set the model
    variogram.model = params['model']

    # set the parameters
    if params.get('shape') is not None:
        variogram.fit(
            range=params.get('range'),
            sill=params.get('sill'),
            nugget=params.get('nugget'),
            shape=params.get('shape')
        )
    else:
        variogram.fit(
            range=params.get('range'),
            sill=params.get('sill'),
            nugget=params.get('nugget')
        )

    return variogram


def parameterized_clone(variogram: Variogram, parameters: Union[List[float], dict]) -> Variogram:
    """
    Creates a clone of variogram with the given parameters already fitted.
    """
    # check parameters
    if not isinstance(parameters, dict):
        if variogram.model.__name__ in ('stable', 'matern'):
            parameters = {
                'range': parameters[0],
                'sill': parameters[1],
                'nugget': parameters[3],
                'shape': parameters[2]
            }
        else:
            parameters = {
                'range': parameters[0],
                'sill': parameters[1],
                'nugget': parameters[2]
            }
    
    # re-build the variogram
    vario = variogram.clone()
    vario.fit(method='manual', **parameters)

    return vario
