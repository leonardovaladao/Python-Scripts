from .linear_algebra import Vector, sum_of_squares, sqrt, dot
from collections import Counter
from typing import List, TypeVar, Callable
from random import choice

X = TypeVar("X")
Stat = TypeVar("Stat")

def mean(v: Vector)->float:
    """ Return mean of vector """
    return sum(v)/len(v)

def median(v: Vector)->float:
    """ Return median of vector """
    if len(v)%2==0:
        sorted_v=sorted(v)
        hi_midpoint=len(v)//2
        return (sorted_v[hi_midpoint-1]+sorted_v[hi_midpoint])/2
    else:
        return sorted(v)[len(v)//2]
    
def quantile(v: Vector, p: float)->float:
    """ Return quantile p of vector v """
    p_index = int(p*len(v))
    return sorted(v)[p_index]

def mode(v: Vector)-> Vector:
    """ Returns the modes of the vector """
    counts = Counter(v)
    max_count = max(counts.values())
    return [xi for xi, count in counts.items() if count==max_count] 

def data_range(v: Vector)->float:
    """ Returns the range of vector """
    return max(v)-min(v)

def de_mean(v: Vector)-> Vector:
    """ Standartize vector by subtracting each element by the vector mean """
    x_mean = mean(v)
    return [x-x_mean for x in v]

def variance(v: Vector)->float:
    """ Compute the vector variance """
    assert len(v)>=2, "Vector length must be greater than 2"
    n=len(v)
    deviations=de_mean(v)
    return sum_of_squares(deviations)/(n-1)

def standard_deviation(v: Vector)->float:
    """ Compute the standard deviation of vector """
    return sqrt(variance(v))

def interquartile_range(v: Vector)->float:
    """ Returns the difference between 75th and 25th quantiles """
    return quantile(v, 0.75)-quantile(v, 0.25)

def covariance(v: Vector, w: Vector)->float:
    """ Compute covariance between two vectors """
    return dot(de_mean(v), de_mean(w)) / len(v)-1

def correlation(v: Vector, w: Vector)->float:
    """ Compute correlation between two vectors """
    std_v = standard_deviation(v)
    std_w = standard_deviation(w)
    return covariance(v,w) / std_v / std_w

def bootstrap_sample(data: List[X])->List[X]:
    """ Randomly samples len(data) elements with replacement """
    return [choice(data) for _ in data]

def bootstrap_statistic(data: List[X], stats_fn: Callable[[List[X]], Stat],
                        num_samples:int)-> List[Stat]:
    """ Evaluates stats_fn on num_samples bootstrap samples from data """
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

