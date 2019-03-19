import numpy as np
import time

def timer(name):
    def timer_decorator(func):
        def timed_fun(*args, **kwargs):
            start = time.time()
            return_values = func(*args, **kwargs)
            end = time.time()
            print("\n Elapsed time for ", name, " : ",  end - start, "\n")
            return return_values
        return timed_fun
    return timer_decorator

def info(func):
    def wrapped_fun(*args, **kwargs):
        print("\nRunning ", func.__name__, " .... ")
        return_values = func(*args, **kwargs)
        print("Completed ", func.__name__, "")
        return return_values
    return wrapped_fun

def reset(func):
    def wrapped_fun(*args, **kwargs):
        return_values = func(*args, **kwargs)
        return return_values
    return wrapped_fun


def mae(x, y):
    return np.mean(np.abs(x - y))
