from random import shuffle
from typing import List, Tuple, TypeVar

X = TypeVar("X") #generic type

def split_data(data: List[X], prob:float)->Tuple[List[X], List[X]]:
    """ Split data into fractions [prob, 1-prob] """
    data = data[:]
    shuffle(data)
    cut=int(len(data)*prob)
    return data[:cut], data[cut:]

Y = TypeVar("Y")
def train_test_split(xs: List[X], ys: List[Y], test_size:float)->Tuple[List[X], List[X], List[Y], List[Y]]:
    """ Returns randomly splitted data. Example:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 0.75) """
        
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1-test_size)
    
    return ([xs[i] for i in train_idxs], # x_train
            [xs[i] for i in test_idxs], # x_test
            [ys[i] for i in train_idxs], # y_train
            [ys[i] for i in test_idxs]  # y_test
            )

def accuracy(tp: int, fp: int, fn: int, tn: int)->float:
    correct = tp+tn
    total = tp+fp+fn+tn
    return correct/total

def precision(tp:int, fp:int, fn:int, tn: int)->float:
    return tp/(tp+fp)

def recall(tp:int, fp: int, fn: int, tn:int)->float:
    return tp/(tp+fn)

def f1_score(tp:int, fp:int, fn:int, tn:int)->float:
    p=precision(tp,fp,fn,tn)
    r=recall(tp,fp,fn,tn)
    return 2*p*r/(p+r)