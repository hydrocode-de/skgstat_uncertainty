import abc
from typing import Union
from sympy import Symbol, IndexedBase, diff, Integer, summation

class ExactEstimator(abc.ABC):
    """Abstract Base class
    """
    # define symbols to solve the variogram estimator formulae 
    i = Symbol('i', positive=True, integer=True)
    N = Symbol('N', positive=True, integer=True)
    X  = IndexedBase('x')

    @abc.abstractmethod
    def estimator(self):
        """
        Symbolic estimator formula
        """
        pass

    def derivative(self):
        """
        Calculate the derivate of the the estimator function
        with respect to x_i
        """
        return diff(self.estimator(), self.X[self.i])

    def __init__(self, data: Union[list, tuple] = None):
        self._model = self.estimator()

        if data is not None:
            self.init_with_data(data)

    def init_with_data(self, data: Union[list, tuple]):
        # make a raw copy of the data
        self.data = data
        self.n = len(data)

        # set my model with data and its derivative
        self._model = self.estimator().subs(self.N, self.n - 1)
        self._diff = diff(self.model, self.X[self.i])

    @property
    def model(self):
        return self._model

    @property
    def deriv(self):
        return self._diff

    def solve_deriv(self, evalf: bool = False):
        """
        Solve the derivative of the model with respect to x_i,
        substitite the symbols with the actual data of this instance.
        """
        # substitute each element of x with data
        res = self.deriv.doit().subs([(self.X[i], self.data[i]) for i in range(self.n)])

        # return as symbolic or approimate numerically
        if evalf:
            return res.evalf()
        else:
            return res

    def __call__(self, data: Union[list, tuple] = None, evalf: bool = False):
        # replace data if needed
        if data is not None:
            self.init_with_data(data)
        
        return self.solve_deriv(evalf)


class Matheron(ExactEstimator):
    """
    Matheron's estimator
    """
    def estimator(self):
        return (Integer(1) / (2 * self.N)) * summation(self.X[self.i] ** 2, (self.i, 0, self.N))