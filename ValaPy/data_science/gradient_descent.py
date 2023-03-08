from typing import Callable, List, TypeVar, List, Iterator
from .linear_algebra import add, scalar_multiply
from random import shuffle

Vector = List[float]
T = TypeVar('T')

def difference_quotient(f: Callable[[float], float], 
                        x: float, h: float)->float:
    """ Compute the differentiation quotient of function f with value x and small change h. \n
    In other words, compute the slope of tangent line in function x, at point (x+h, f(x+h)). """
    return (
        ( f(x+h)-f(x) ) / h
        )

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector, i: int, h:float)->float:
    """ Returns i-th partial difference quotiente of f at v """
    w = [vj + (h if j==i else 0) for j, vj in enumerate(v)]
    
    return ((f(w)-f(v))/h)

def estimate_gradient(f: Callable[[Vector], float], 
                      v:Vector, h:float=0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float)->Vector:
    """ Moves step_size in the gradient direction from v """
    assert len(v)==len(gradient)
    step=scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector)-> Vector:
    return [2*vi for vi in v]

def linear_gradient(x: float, y: float, theta: Vector)->Vector:
    slope, intercept = theta
    predicted = slope*x+intercept
    error = (predicted-y)
    grad = [2*error*x, 2*error]
    return grad

def minibatches(dataset: List[T], batch_size: int, shuffles: bool=True)->Iterator[List[T]]:
    """ Generates batch_size-sized minibatches from the dataset """
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffles: shuffle(batch_starts)
    for start in batch_starts:
        end = start+batch_size
        yield dataset[start:end]