"""
The SciKit-GSTat Uncertainty package contains two methods to sample a field dataset
from the database. The result can be used to build a new dataset of internal 'sample'
type. Right now, samples are the only datasets that can be consumed by the variogram
estimation tools. If you want to feed a full field into a variogram, you need to 
exsaustively sample the field. This can be achieved by performing a grid sampling
and passing the field dimensions as desired output grid.


"""
from typing import Tuple, List
import numpy as np


def random(field: List[list], N: int, seed: int = None) -> Tuple[List[list], list]:
    """
    Random sample of the given field by taking N permutations of the coordinate 
    meshgrid.

    Parameters
    ----------
    field : list
        The field can be given as a list of equal size lists containing 
        numeric field values, or as a numpy array.
    N : int
        sample size
    seed : int
        Seed for the random number generator

    Returns
    -------
    coords : list
        List of 2D coordinates (sample locations)
    values : list
        List of observation quatities

    """
    # turn the field into a numpy array
    arr = np.array(field)

    # build a meshgrid over all coordinate dimensions
    grid = np.meshgrid(*[range(dim) for dim in arr.shape])
    mesh = list(zip(*[axis.flatten() for axis in grid]))

    # take N permutations
    rng = np.random.default_rng(seed)
    coords = rng.choice(mesh, size=N, replace=False).tolist()

    # get the values
    values = [arr[tuple(c)].item() for c in coords]

    return coords, values


def grid(field: List[list], N: int = None, spacing: List[int] = None, shape: List[int] = None, offset: List[int] = None) -> Tuple[List[list], list]:
    """
    Sample a field on a regular grid. Yon can either specify the spacing of the target grid
    or specify its dimensions to auto-calculate the spacing. Both options can be offset from
    the boundaries.

    Parameters
    ----------
    field : list
        The field can be given as a list of equal size lists containing 
        numeric field values, or as a numpy array.
    N : int
        You can pass the number of observations the target grid should have in total.
        The tool tries to figure out a regular grid dimension, that holds N observations.
        .. note::
            It is possible that the sample size is larger or smaller than N in case
            N cannot be distributed along a regular grid.
        
    spacing : list
        The spacing list has to be of field.ndim size.
        Each number is the spacing in grid units along the respective axis.
        Will be ignored if N is given.
    shape : list
        The shape list has to be of field.ndim size.
        Each number specifies the desired target cells along the respective axis.
        Will be ignored if N or spacing is given.
    offset : list
        The offset list has to be of field.ndim size. 
        Each number specifies the offset from the boundary in grid units along
        the respective axis. Can be combined with N, spacing and shape.

    Returns
    -------
    coords : list
        List of 2D coordinates (sample locations)
    values : list
        List of observation quatities

    """
    # turn the field into a numpy array
    arr = np.array(field)

    if spacing is None and N is None and shape is None:
        raise AttributeError('Either N, spacing or shape has to be given.')
    
    if offset is None:
        offset = [0 for _ in range(arr.ndim)]
    elif isinstance(offset, int):
        offset = [offset for _ in range(arr.ndim)]

    # if N is given, derive the grid from N
    if N is not None:
        # build the grid by linspaceing each dimension in matrix coordinates
        grid = np.meshgrid(*[np.linspace(off, dim - 1 - off, int(np.power(N, 1 / arr.ndim))) for dim, off in zip(arr.shape, offset)])

    elif spacing is not None:
        # arange the spacing along each axis
        grid = np.meshgrid(*[np.arange(off, dim - 1 - off, s) for s, dim, off in zip(spacing, arr.shape, offset)])

    elif shape is not None:
        # linspace the desired shape along each axis
        grid = np.meshgrid(*[np.linspace(off, dim - 1 - off, s) for s, dim, off in zip(shape, arr.shape, offset)])

    #create the coordinates
    coords = np.asarray(list(zip(*[axis.flatten() for axis in grid])), dtype=int).tolist()

    # sample the field
    values = [arr[tuple(c)].item() for c in coords]

    return coords, values


def transect(field: List[list], p1: Tuple[int, int], p2: Tuple[int, int], N: int = None, spacing: int = None) -> Tuple[List[list], list]:
    # turn the field into a numpy array
    arr = np.array(field)

    if spacing is None and N is None:
        raise AttributeError('Either N or spacing has to be given')

    # calculate N from spacing
    if spacing is not None:
        N = int(np.sqrt(np.sum(np.power(p2 - p1, 2))) / spacing)

    # create the sampling coordinates along the transect
    if N is not None:
        x = np.linspace(p1[0], p2[0], N).astype(int)
        y = np.linspace(p1[1], p2[1], N).astype(int)
    
    coords = list(zip(x, y))
    values = arr[x, y].tolist()

    return coords, values


