from math import exp, log
from .linear_algebra import Vector, dot, vector_sum
from .gradient_descent import gradient_step
from typing import List
from random import random
from tqdm import trange

class Logistic_Regression():
    def __init__(self):
        pass
    
    def logistic_function(self, x:float)->float:
        return 1.0/(1+exp(-x))
    
    def logistic_derivative(self, x:float)->float:
        y=self.logistic_function(x)
        return y*(1-y)
    
    def _negative_log_likelihood_dp(self, x: Vector, y: float, 
                                    beta: Vector)->float:
        if y==1:
            return -log(self.logistic_function(dot(x,beta)))
        else:
            return -log(1-self.logistic_function(dot(x,beta)))
        
    def negative_log_likelihood(self, xs: List[Vector], ys: List[float], 
                                beta: Vector) -> Vector:
        return sum(self._negative_log_likelihood_dp(x, y, beta) for x,y in zip(xs,ys))
    
    def _negative_log_partial_j(self, x: Vector, y: float, beta: Vector, j: int) -> float:
        """
        The jth partial derivative for one data point.
        Here i is the index of the data point.
        """
        return -(y - self.logistic_function(dot(x, beta))) * x[j]
    
    
    def _negative_log_gradient(self, x: Vector, y: float, beta: Vector) -> Vector:
        """
        The gradient for one data point.
        """
        return [self._negative_log_partial_j(x, y, beta, j)
                    for j in range(len(beta))]
    
    
    def negative_log_gradient(self, xs: List[Vector],
                              ys: List[float],
                              beta: Vector) -> Vector:
        return vector_sum([self._negative_log_gradient(x, y, beta)
                           for x, y in zip(xs, ys)])
    
    def fit(self, xs: List[Vector], ys: List[float],
            seed:int=42,
            learning_rate:float=0.01,
            num_epochs:int=5000):
        
        lenx = len(xs[0])
        beta = [random() for i in range(lenx)]
        
        with trange(num_epochs) as t:
            for epoch in t:
                gradient = self.negative_log_gradient(xs, ys, beta)
                beta = gradient_step(beta, gradient, -learning_rate)
                loss = self.negative_log_likelihood(xs, ys, beta)
                t.set_description(f"loss: {loss:.3f}")
                
        self.beta = beta               
                
        return None
    
    def predict_probs(self, xs: List[Vector])->List[float]:
        return [self.logistic_function(dot(self.beta, xi)) for xi in xs]
    
    def predict(self, xs: List[Vector])->List[int]:
        probs = self.predict_probs(xs)
        return [0 if i<0.5 else 1 for i in probs]
    
    def accuracy(self, ys_true: List[float], ys_pred: List[float])->float:
        return sum([1 if i==j else 0 for i,j in zip(ys_true,ys_pred)])/len(ys_true)
    
    def recall(self, ys_true: List[float], ys_pred: List[float])->float:
        tp = [1 if (i==1 and j==1) else 0 for i,j in zip(ys_true,ys_pred)]
        fn = [1 if (p==0 and t==1) else 0 for t,p in zip(ys_true, ys_pred)]
        return sum(tp)/(sum(tp)+sum(fn))
    
    def precision(self, ys_true: List[float], ys_pred: List[float])->float:
        tp = [1 if (i==1 and j==1) else 0 for i,j in zip(ys_true,ys_pred)]
        fp = [1 if (p==1 and t==0) else 0 for t,p in zip(ys_true, ys_pred)]
        return sum(tp)/(sum(tp)+sum(fp))
    
    def f1_score(self, ys_true: List[float], ys_pred: List[float])->float:
        precision = self.precision(ys_true, ys_pred)
        recall = self.recall(ys_true, ys_pred)
        return (2*precision*recall)/(precision+recall)