from .linear_algebra import dot, Vector, vector_mean, add
from .gradient_descent import gradient_step
from .base_stat import de_mean
from typing import List
from random import random
from tqdm import trange

class Linear_Regression():
    def __init__(self):
        pass
    
    def predict_values(self, x: Vector, beta: Vector)->float:
        """ Assume first element of x is 1 """
        return dot(x, beta)
    
    def error(self, x: Vector, y: float, beta: Vector)-> float:
        return self.predict_values(x,beta)-y

    def squared_error(self, x: Vector, y: float, beta: Vector)->float:
        return self.error(x,y,beta)**2

    def sqerror_gradient(self, x: Vector, y: float, beta: Vector)->Vector:
        err = self.error(x,y, beta)
        return [2*err*xi for xi in x]
    
    def least_squares_fit(self, xs: List[Vector], ys: List[float],
                          learning_rate:float=0.001,
                          num_steps:int=1000,
                          batch_size:int=1)->Vector:
        self.xs = xs
        self.ys = ys
        guess = [random() for _ in xs[0]]
        
        for _ in trange(num_steps, desc='Least Squares Fit'):
            for start in range(0, len(xs), batch_size):
                batch_xs = xs[start:start+batch_size]
                batch_ys = ys[start:start+batch_size]
                
                gradient = vector_mean([self.sqerror_gradient(x,y,guess)
                                        for x,y in zip(batch_xs, batch_ys)])
                guess = gradient_step(guess, gradient, -learning_rate)
        self.beta = guess
        return self.beta
    
    def total_sum_of_squares(self, y: Vector)->float:
        """ Total squared variation of y_i from mean """
        return sum(v**2 for v in de_mean(y))
    
    def r2(self):
        sum_of_sqerrors = sum(self.error(x, y, self.beta)**2 
                              for x,y in zip(self.xs, self.ys))
        return 1.0-sum_of_sqerrors/self.total_sum_of_squares(self.ys)
    
    def ridge_penalty(self, beta: Vector, alpha: float)->float:
        return alpha*dot(beta[1:], beta[1:])
    
    def sqerror_ridge(self, x: Vector, y: float, beta: Vector, alpha: float)->float:
        return self.error(x,y,beta)**2+self.ridge_penalty(beta, alpha)
    
    def ridge_penalty_gradient(self, beta: Vector, alpha: float)->Vector:
        return [0.]+[2*alpha*betaj for betaj in beta[1:]]
    
    def sqerror_ridge_gradient(self, x: Vector, y: float, beta: Vector, alpha: float)->Vector:
        return add(self.sqerror_gradient(x,y,beta), self.ridge_penalty_gradient(beta, alpha))
    
    def fit(self, xs: List[Vector], ys: List[float],
            alpha:float=0.0,
            learning_rate:float=0.001,
            num_steps:int=1000,
            batch_size:int=1)->Vector:
        self.xs = xs
        self.ys = ys
        guess = [random() for _ in xs[0]]
        
        for _ in trange(num_steps, desc='Ridge Loss Fit'):
            for start in range(0, len(xs), batch_size):
                batch_xs = xs[start:start+batch_size]
                batch_ys = ys[start:start+batch_size]
                
                gradient = vector_mean([self.sqerror_ridge_gradient(x,y,guess, alpha)
                                        for x,y in zip(batch_xs, batch_ys)])
                guess = gradient_step(guess, gradient, -learning_rate)
        self.beta = guess
        return None
        
    def coefs(self):
        return self.beta
    
    def predict(self, x:Vector)->Vector:
        return [self.predict_values(xi, self.beta) for xi in x]