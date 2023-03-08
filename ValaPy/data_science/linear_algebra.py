from typing import List, Tuple, Callable
from math import sqrt

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """ Adds two vectors """
    assert len(v)==len(w), 'Vectors v and w must be of same length'
    return [vi+wi for vi,wi in zip(v,w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """ Subtracts two vectors """
    assert len(v)==len(w), 'Vectors v and w must be of same length'
    return [vi-wi for vi,wi in zip(v,w)]

def vector_sum(vectors: List[Vector])->Vector:
    """ Pass a list of vectors and returns the sum of them' """ 
    assert vectors, 'Please pass a list of vectors'
    
    num_elements=len(vectors[0])
    assert all(len(v)==num_elements for v in vectors), 'All vectors must be the same size'
    
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(s: float, v: Vector)->Vector:
    """ Multiply a scalar s by a vector v' """ 
    return [s*vi for vi in v]

def vector_mean(vectors: List[Vector])->Vector:
    """ Compute the mean vector """ 
    n=len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v: Vector, w: Vector)->float:
    """ Compute the dot product of two vectors' """ 
    assert len(v)==len(w), 'Vectors must be the same length'
    return sum(vi*wi for vi,wi in zip(v,w))

def sum_of_squares(v: Vector)->float:
    """ Return dot product of vector with itself """ 
    return dot(v,v)

def magnitude(v:Vector)->float:
    """ Returns magnitude of vector """ 
    return sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector)->float:
    """ Compute the squared distance between two vectors """ 
    return sum_of_squares(subtract(v,w))

def distance(v: Vector, w: Vector)->float:
    """ Compute the distance between two vectors """ 
    return sqrt(squared_distance(v, w))

Matrix = List[Vector]

def shape(A:Matrix)->Tuple[int, int]:
    """ Returns the shape of matrix """ 
    num_rows=len(A)
    num_cols=len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A:Matrix, i:int)->Vector:
    """ Get ith row of matrix """ 
    return A[i]

def get_col(A:Matrix, j:int)->Vector:
    """ Get jth row of matrix """ 
    return [Ai[j] for Ai in A]

def make_matrix(n_rows:int, n_cols:int, 
                function: Callable[[int, int], float]) -> Matrix:
    """ Returns a n_rowx x n_cols matrix based on function """
    return [[function(i,j) 
             for j in range(n_cols)]
            for i in range(n_rows)]

def identity_matrix(n:int)->Matrix:
    """ Returns n x n identity matrix """
    return make_matrix(n,n,lambda i,j: 1 if i==j else 0)

def print_matrix(A: Matrix)->None:
    for line in A:
        print(line)

