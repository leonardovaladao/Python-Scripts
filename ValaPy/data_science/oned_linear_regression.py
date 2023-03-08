from .linear_algebra import Vector
from .base_stat import correlation, standard_deviation, mean, de_mean
from .gradient_descent import gradient_step
from typing import Tuple
import random
from tqdm import trange
from math import sqrt

class Simple_Linear_Regression():
    def __init__(self):
        pass
    
    def predict_value(self, alpha:float, beta:float, xi:float)->float:
        """ Predict value based on linear regression. """    
        return beta*xi+alpha
    
    def error(self, alpha:float, beta:float, xi:float, yi:float)->float:
        """ Compute error from predicted value based on linear regression """
        return self.predict_value(alpha,beta,xi)-yi
    
    def sum_of_sqerrors(self, alpha:float, beta:float, x: Vector, y: Vector)->float:
        return sum(self.error(alpha, beta, xi, yi)**2 for xi, yi in zip(x,y))
    
    def least_squares_fit(self, x: Vector, y: Vector)->Tuple[float,float]:
        """ Given vectors x and y, find least-squared values of alpha and beta """
        assert type(x[0])!=list, "Make sure your independent variable has one dimension. For multi-dimensional independent variables, use the Linear_Regression() class."
        
        self.X = x
        self.Y = y
        self.beta = correlation(x,y) * standard_deviation(y) / standard_deviation(x)
        self.alpha = mean(y)-self.beta*mean(x)
        return self.alpha, self.beta
    
    def total_sum_of_squares(self, y: Vector)->float:
        """ Total squared variation of y_i from mean """
        return sum(v**2 for v in de_mean(y))
    
    def r2(self)->float:
        return 1.0- (self.sum_of_sqerrors(self.alpha, self.beta, self.x, self.y)/self.total_sum_of_squares(self.y))
    
    def fit(self, x:Vector, y:Vector, num_epochs = 10000, seed=0, learning_rate = 0.00001):
        assert type(x[0])!=list, "Make sure your independent variable has one dimension. For multi-dimensional independent variables, use the Linear_Regression() class."
        self.x = x
        self.y = y
        
        random.seed(seed)
        guess = [random.random(), random.random()] # choose random value to start
        
        with trange(num_epochs) as t:
            for _ in t:
                alpha, beta = guess
                
                grad_a = sum(2 * self.error(alpha, beta, x_i, y_i)
                             for x_i, y_i in zip(x,
                                                 y))
            
                grad_b = sum(2 * self.error(alpha, beta, x_i, y_i) * x_i
                             for x_i, y_i in zip(x,
                                                 y))
            
                loss = self.sum_of_sqerrors(alpha, beta, x, y)
                t.set_description(f"loss: {loss:.3f}")
            
                guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
            
        self.alpha, self.beta = guess
        
        return None
    
    def coefs(self):
        return self.alpha, self.beta
    
    def predict(self, x:Vector)->Vector:
        return [self.predict_value(self.alpha, self.beta, xi) for xi in x]
        
    def mse(self, y_real: Vector, y_pred: Vector)->float:
        return sqrt( sum( [(yi-yih)**2 for yi, yih in zip(y_real, y_pred)] ) /len(y_real) )