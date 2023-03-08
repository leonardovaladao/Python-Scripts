from data_science import *
import random

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

outlier = num_friends.index(100)    # index of outlier
num_friends_good = [x
                    for i, x in enumerate(num_friends)
                    if i != outlier]
daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]


import matplotlib.pyplot as plt
from numpy import arange

def test_gaussian():
    x = arange(-4, 4, 0.01)
    y = [normal_pdf(xi) for xi in x]
    y2 = [normal_pdf(xi, 0, 2) for xi in x]
    y3 = [normal_pdf(xi, 0, 0.5) for xi in x]
    y4 = [normal_pdf(xi, -1, 1) for xi in x]
    
    plt.plot(x, y, label='mu=0, sigma=1')
    plt.plot(x, y2, label='mu=0, sigma=2')
    plt.plot(x, y3, label='mu=0, sigma=0.5')
    plt.plot(x, y4, label='mu=-1, sigma=1')
    plt.legend()
    plt.show()
    
def test_gaussian_cdf():
    x = arange(-4, 4, 0.01)
    y = [normal_cdf(xi) for xi in x]
    y2 = [normal_cdf(xi, 0, 2) for xi in x]
    y3 = [normal_cdf(xi, 0, 0.5) for xi in x]
    y4 = [normal_cdf(xi, -1, 1) for xi in x]
    
    plt.plot(x, y, label='mu=0, sigma=1')
    plt.plot(x, y2, label='mu=0, sigma=2')
    plt.plot(x, y3, label='mu=0, sigma=0.5')
    plt.plot(x, y4, label='mu=-1, sigma=1')
    plt.legend()
    plt.show()
    
def test_gradient_step_size():
    v = [random.uniform(-10, 10) for i in range(3)]
    for epoch in range(1000):
        grad = sum_of_squares_gradient(v) # compute the gradient at v
        v = gradient_step(v, grad, -0.01) # take a negative gradient step
        print(epoch, v)
    print(distance(v, [0,0,0]))

def test_linear_grad():
    inputs = [(x, 20*x+5) for x in range(-50,50)]
    theta = [random.uniform(-1,1), random.uniform(-1, 1)]
    learning_rate=0.001
    
    for epoch in range(5000):
        grad = vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    
    plt.plot([i[0] for i in inputs], [i[1] for i in inputs], label='original')
    plt.plot([i[0] for i in inputs],
             [intercept+i[0]*slope for i in inputs], label='estimate')
    plt.legend()
    plt.show()
        
def test_linear_grad_w_batch():
     inputs = [(x, 20*x+5) for x in range(-50,50)]
     theta = [random.uniform(-1,1), random.uniform(-1, 1)]
     learning_rate=0.001
     

     for epoch in range(1000):
         for batch in minibatches(inputs, batch_size=20):
             grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
             theta = gradient_step(theta, grad, -learning_rate)
             print(epoch, theta)
     slope, intercept=theta
     
     plt.plot([i[0] for i in inputs], [i[1] for i in inputs], label='original', ls=':')
     plt.plot([i[0] for i in inputs],
              [intercept+i[0]*slope for i in inputs], label='estimate', linestyle=':')
     plt.legend()
     plt.show()
     
def test_linear_grad_stochastic():
    inputs = [(x, 20*x+5) for x in range(-50,50)]
    theta = [random.uniform(-1,1), random.uniform(-1, 1)]
    learning_rate=0.001
    
    for epoch in range(100):
        for x,y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    slope,intercept=theta
    
    plt.plot([i[0] for i in inputs], [i[1] for i in inputs], label='original', ls=':')
    plt.plot([i[0] for i in inputs],
             [intercept+i[0]*slope for i in inputs], label='estimate', linestyle=':')
    plt.legend()
    plt.show()
             
def test_histogram():
    
    random.seed(42)
    uniform=[200*random.random()-100 for _ in range(10000)]
    normal = [57*inverse_normal_cdf(random.random()) for _ in range(10000)]
    
    plot_histogram(uniform, 10, 'Uniform Histogram')
    
def test_rescale():    
    vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
    print(scale(vectors))
    print(scale(rescale(vectors)))
    
def split():
    data = [n for n in range(1000)]
    train, test = split_data(data, 0.75)
    print(len(train), len(test))
    
def train_test_split_test():
    xs = [x for x in range(1000)] # xs are 1 ... 1000
    ys = [2 * x for x in xs] # each y_i is twice x_i
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)
    print(len(x_train), len(x_test), len(y_train), len(x_test))
    
def test_least_square_fit():
    outlier = num_friends.index(100)    # index of outlier
    num_friends_good = [x
                        for i, x in enumerate(num_friends)
                        if i != outlier]
    daily_minutes_good = [x
                          for i, x in enumerate(daily_minutes)
                          if i != outlier]    
    X = num_friends_good
    Y = daily_minutes_good
    
    slr = Simple_Linear_Regression()
    
    slr.least_squares_fit(num_friends_good, daily_minutes_good)
    
    print(slr.r2())
    
def test_simple_linear_regression():
    
    
    slr = Simple_Linear_Regression()
    slr.fit(num_friends_good, daily_minutes_good, num_epochs=10000)
    
    alpha, beta = slr.coefs()
    
    assert 22.9 < alpha < 23.0
    assert 0.9 < beta < 0.905
    
    y_pred = slr.predict(num_friends_good)
    
    plt.scatter(num_friends_good, daily_minutes_good)
    plt.plot(num_friends_good, y_pred, color='red')
    
    mse = slr.mse(daily_minutes_good, y_pred)
    print(sqrt(mse))
    print(slr.r2())
    
def test_multi_lr():
    lr = Linear_Regression()
    x = [1,2,3]
    y = 30
    beta = [4,4,4]

    beta = lr.fit(inputs, daily_minutes_good, num_steps=5000, batch_size=25)
    print(lr.r2())


def test_log_reg():
    tuples = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
    data = [list(row) for row in tuples]

    xs = [[1.0] + row[:2] for row in data] 
    ys = [row[2] for row in data]
    
    plt.scatter([xi[1] for xi in xs],ys, label='experience', alpha=0.5)
    plt.legend()
    plt.show()
    plt.scatter([xi[2] for xi in xs],ys, label='salary', alpha=0.5)
    plt.legend()
    plt.show()
    
    from data_science.machine_learning import train_test_split
    from data_science.data_handler import rescale
    import tqdm
    rescaled_xs = rescale(xs)
    x_train, x_test, y_train, y_test = train_test_split(rescaled_xs, ys, 0.3)
  
    logreg = Logistic_Regression()
    
    logreg.fit(rescaled_xs, ys)
    
    y_pred = logreg.predict(x_test)

    plt.scatter([xi[2] for xi in x_test], y_test, label='Real', alpha=0.5)
    plt.scatter([xi[2] for xi in x_test], y_pred, label='Predicted', color='red', alpha=0.5 )
    plt.legend()
    plt.show()
    
    print(y_pred)
    
def t_train_test_split():
    from data_science.machine_learning import train_test_split

    x = [i for i in range(100)]
    y = [i*2+5 for i in x]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.1)
    
    print(len(x_train), len(y_train), '\n')
    print(len(x_test), len(y_test))
      
def decision_tree_test():
    dt = Decision_Tree()
    
    from typing import NamedTuple, Optional
    class Candidate(NamedTuple):
        level: str
        lang: str
        tweets: bool
        phd: bool
        did_well: Optional[bool] = None  # allow unlabeled data
    
                      #  level     lang     tweets  phd  did_well
    inputs = [Candidate('Senior', 'Java',   False, False, False),
              Candidate('Senior', 'Java',   False, True,  False),
              Candidate('Mid',    'Python', False, False, True),
              Candidate('Junior', 'Python', False, False, True),
              Candidate('Junior', 'R',      True,  False, True),
              Candidate('Junior', 'R',      True,  True,  False),
              Candidate('Mid',    'R',      True,  True,  True),
              Candidate('Senior', 'Python', False, False, False),
              Candidate('Senior', 'R',      True,  False, True),
              Candidate('Junior', 'Python', True,  False, True),
              Candidate('Senior', 'Python', True,  True,  True),
              Candidate('Mid',    'Python', False, True,  True),
              Candidate('Mid',    'Java',   True,  False, True),
              Candidate('Junior', 'Python', False, True,  False)
             ]
    """
    for key in ['level', 'lang', 'tweets', 'phd']:
        print(key, dt.partition_entropy_by(inputs, key, 'did_well'))

    senior_inputs = [input for input in inputs if input.level=='Senior']
    print('\n',dt.partition_entropy_by(senior_inputs, 'lang', 'did_well'))
    """
    tree = dt.build_tree_id3(inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well')
    print(dt.classify(tree, Candidate("Junior", "Java", True, False)))
    print(dt.classify(tree, Candidate("Senior", "R", False, False)))

def test_knn():
    import requests
    data = requests.get(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        )
    with open('iris.dat', 'w') as f:
        f.write(data.text)

test_knn()