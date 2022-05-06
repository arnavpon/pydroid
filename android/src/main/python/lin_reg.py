import numpy as np
from sklearn.linear_model import LinearRegression
import time

def generate_random_nparray(n):
  # generates an n x 1 array of random numbers for regression
  rand_array = np.random.rand(n)
  return rand_array

def model(size):
  # simple linear regression model for performance testing
  start = time.perf_counter()  # start timer
  x = generate_random_nparray(size).reshape(-1, 1)
  y = generate_random_nparray(size)

  model = LinearRegression()
  model.fit(x, y)
  rsquare = model.score(x, y)
  # print("R^2: {0} | Intercept: {1} | Coefficient: {2}".format(rsquare, model.intercept_, model.coef_))
  stop = time.perf_counter()  # end timer
  runtime = stop - start
  # print(f"Time Elapsed: {runtime:0.4f} seconds")
  return runtime
  

def average_performance(n, m_size):
  # runs the linear regression n times on a model of size m_size & gets avg time 
  runtimes = []
  for i in range(n):
    runtimes.append(model(m_size))
  runtime_array = np.array(runtimes)
  average = runtime_array.sum() / len(runtime_array)
  print(f"{average:0.4f}")

average_performance(100, 1000000)