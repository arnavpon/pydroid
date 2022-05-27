import numpy as np
from sklearn.linear_model import LinearRegression
import time
import json

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
  runtime = (stop - start) * 1000 
  # print(f"Time Elapsed: {runtime:0.4f} milliseconds")
  return runtime
  

def average_performance_old(n, m_size):
  # runs the linear regression n times on a model of size m_size & gets avg time 
  print(f"[python] Running {n} model iterations with {m_size} values each")
  runtimes = []
  for i in range(n):
    runtimes.append(model(m_size))
  runtime_array = np.array(runtimes)
  average = runtime_array.sum() / len(runtime_array)
  print(f"[python] Average runtime: {average:0.4f}")
  return average

def main(arguments):
  print("[Python] Main function of lin_reg()...")
  # runs the linear regression n times on a model of size m_size & gets avg time 
  n = arguments.get("iterations")
  m_size = arguments.get("model_size")
  print(f"[python] Running {n} model iterations with {m_size} values each")
  runtimes = []
  for i in range(n):
    runtimes.append(model(m_size))
  runtime_array = np.array(runtimes)
  average = runtime_array.sum() / len(runtime_array)
  print(f"[python] Average runtime: {average:0.4f}")
  res_as_json = json.dumps(average)  # convert to JSON before returning because we can't pass generic Object type -> dart through java, so we need to use a known primitive; the most flexible of these is a string representing JSON
  print(res_as_json, type(res_as_json))
  return res_as_json  

