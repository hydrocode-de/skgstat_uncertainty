from typing import List, Tuple

from skgstat import Variogram
import numpy as np
from scipy import stats


def residual_uncertainty(variogram: Variogram, q: List[int] = [5, 95]) -> List[Tuple[float, float]]:
    # get the estimator function
    estimator = variogram.estimator
    
    def calc(residuals: np.array) -> Tuple[float, float]:
        if len(residuals) == 0:
            return np.nan, np.nan
        
        # get the lower and higher confidence interval
        lo = np.percentile(residuals, q=q[0])
        up = np.percentile(residuals, q=q[1])
        
        # find the residuals within the confidence interval
        lower = residuals[np.where(residuals >= lo)]
        upper = residuals[np.where(residuals <= up)]
        
        # recalculate the estimator
        return estimator(lower), estimator(upper)
    
    # map recalc
    return [calc(r) for r in variogram.lag_classes()]


def kfold_residual_bootstrap(variogram: Variogram, k: int = 5, repititions: int = 100, q: List[int] = [5, 95], seed: int = None) -> List[Tuple[float, float]]:
    # get the estimator
    estimator = variogram.estimator
    
    # get a rng
    rng = np.random.default_rng(seed)

    def calc(residuals: np.ndarray) -> Tuple[float, float]:
        if len(residuals) < k:
            return np.nan, np.nan

        # create a container for the estimates of all folds
        estimates = []

        # go for all repitions
        for _ in range(repititions):
            # copy the current lag class
            res = residuals.copy()

            # shuffle randomly
            rng.shuffle(res)

            # create the folds
            folds = np.split(res, [int((i / k) * len(res)) for i in range(1, k)])

            # create the estimates for the k folds of this repitions
            e = [estimator(np.concatenate([folds[j] for j in range(k) if j!=i])) for i in range(k)]

            # add to container
            estimates.extend(e)
        
        # cross validation finished, calcuate confidence interval
        return np.percentile(estimates, q[0]), np.percentile(estimates, q[1])

    # map calc to all lag classes
    return [calc(r) for r in variogram.lag_classes()]


def conf_interval_from_sample_std(variogram: Variogram, conf_level: 0.95) -> List[Tuple[float, float]]:
    intervals = []

    # calculate the confidence interval for each group
    for est, grp in zip(variogram.experimental, variogram.lag_classes()):
        # get the z-score for standard normal distribution of given conf interval
        z = stats.norm.ppf(conf_level)

        # get the n and standard deviation of the sample
        n = len(grp)
        s = np.std(grp)
        
        # get the deviation
        dev = z * s / np.sqrt(n)

        # calculate the confidence interval
        intervals.append((est - dev if est - dev > 0 else 0, est + dev))
    return intervals


def _std_input_generator(values: np.ndarray, std: float, length: int, seed: int = None) -> np.ndarray:
    # get a random generator
    rng = np.random.default_rng(seed)

    # create the randomized input
    return rng.normal(loc=values, scale=std, size=(length, values.size))    

def _sem_input_generator(values: np.ndarray, sem: float, length: int, seed: int = None) -> np.ndarray:
    # calculate the standard deviation from SEM
    std = sem * np.sqrt(values.size)

    return _std_input_generator(values, std, length, seed)

def _uniform_input_generator(values: np.ndarray, precision: float, length: int, seed: int = None) -> np.ndarray:
    # get a random generator
    rng = np.random.default_rng(seed)

    deviation = precision / 2

    # create the container
    inputs = np.ones((length, values.size)) * np.nan

    for i, value in enumerate(values):
        inputs[:, i] = rng.uniform(value - deviation, value + deviation, size=length)
    
    return inputs

def  mc_absolute_observation_uncertainty(variogram: Variogram, sigma: float, iterations: int = 500, seed: int = None, return_type='result', sigma_type='sem'):
    # copy the variogram parameters
    params = variogram.describe()['params']
    params.update(fit_method=None)   # disable fitting
    coords = variogram.coordinates

    # cretate the output container
    results = np.ones((iterations, len(variogram.bins))) * np.nan

    # get the inputs
    if sigma_type == 'sem':
        inputs = _sem_input_generator(variogram.values, sigma, iterations, seed)
    elif sigma_type == 'std':
        inputs = _std_input_generator(variogram.values, sigma, iterations, seed)
    elif sigma_type == 'precision':
        inputs = _uniform_input_generator(variogram.values, sigma, iterations, seed)

    # start the simulation
    for i in range(len(inputs)):
        # build a new variogram instance
        vario = Variogram(coords, inputs[i, :], **params)
        
        # store
        results[i] = vario.experimental

        if return_type == 'result':
            yield results
        else:
            yield vario



    
    