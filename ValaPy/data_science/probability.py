from math import exp,sqrt,pi,erf
from random import random

def uniform_pdf(x: float)-> float:
    return 1 if 0 <= x else 0

def uniform_cdf(x:float)-> float:
    """ Compute probability that uniform random variable <= x """
    if x<0: return 0
    elif x<1: return x
    else: return 1
    
def normal_pdf(x:float, mu: float=0, sigma:float=1)->float:
    """ Computes the PDF of gaussian function """
    # f(x|mu,sigma) = 1/(sqrt(2*pi*sigma)) * exp( -(x-mu)**2 / (s*(sigma**2)) )
    return( exp(-(x-mu)**2/2/sigma**2) / sigma*sqrt(2*pi) )

def normal_cdf(x:float, mu:float=0, sigma:float=1)->float:
    """ Return the cumulative distribution function of gaussian function """
    return (1+ erf((x-mu)/sqrt(2)/sigma))/2

def bernoulli_trial(p:float)->int:
    """ Returns 1 with probability p and 0 with probability 1-p """
    return 1 if random()<p else 0

def binomial(n:int, p:float)->int:
    """ Returns the sum of n bernoulli trials """
    return sum(bernoulli_trial(p) for _ in range(n))

def inverse_normal_cdf(p: float, mu: float=0, sigma:float=1, tolerance:float=0.00001)->float:
    """ Find approximate inverse using binary search """
    if mu!=0 or sigma!=1:
        return mu+sigma*inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z=-10.0
    hi_z=10.0
    
    while hi_z-low_z>tolerance:
        mid_z=(low_z+hi_z)/2
        mid_p=normal_cdf(mid_z)
        if mid_p<p:
            low_z=mid_z
        else:
            hi_z=mid_z
            
    return mid_z