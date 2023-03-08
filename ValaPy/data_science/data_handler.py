from math import floor
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import Counter
from .probability import inverse_normal_cdf
from random import random
from .linear_algebra import Matrix, Vector, make_matrix, vector_mean, subtract
from .base_stat import correlation, standard_deviation

def bucketize(point: float, bucket_size: float)->float:
    """ Floor the point to the next lower multiple of bucket_size """
    return bucket_size*floor(point/bucket_size)

def make_histogram(points: List[float], bucket_size: float)->Dict[float, int]:
    """ Buckets the points and counts how many in each bucket """
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()
    
def random_normal()->float:
    """ Returns random from std normal distribution """
    return inverse_normal_cdf(random())

def correlation_matrix(data: List[Vector])->Matrix:
    """ Returns correlation matrix of data """
    def correlation_ij(i:int, j:int)->float:
        return correlation(data[i], data[j])
    return make_matrix(len(data), len(data), correlation_ij)

def scale(data:List[Vector])->Tuple[Vector,Vector]:
    """ Returns mean and stdev for each axis of data. """
    dim=len(data[0])
    means=vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(dim)]
    
    return means,stdevs

def rescale(data:List[Vector])->List[Vector]:
    """ Rescales data so mean=0 and stdev=1. """
    dim=len(data[0])
    means, stdevs=scale(data)
    rescaled = [v[:] for v in data] # copy
    
    for v in rescaled:
        for i in range(dim):
            if stdevs[i]>0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled

def de_mean(data:List[Vector])->List[Vector]:
    """ Recenter data so mean=0 """
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]