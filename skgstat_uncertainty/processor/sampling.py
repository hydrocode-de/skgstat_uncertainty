from typing import Tuple, List
import numpy as np


def random(field: List[list], N: int, seed: int = None) -> Tuple[List[list], list]:
    """
    Random sample of the given field by taking N permutations of the coordinate 
    meshgrid.
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


