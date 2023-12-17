import math
import time
import numpy as np


# file where objective function is declared (dummy test)
def dummy(case):
    time.sleep(0.0001)
    return round(math.sin(sum(case)) * 0.5 + 0.5)

def matmul(case):
    t0 = time.time()
    while(time.time() - t0 < 10):
        m0 = np.random.randint(0, 101, size=(10000, 10000))
        m1 = np.random.randint(0, 101, size=(10000, 10000))
        x = np.dot(m0, m1)
    return round(math.sin(sum(case)) * 0.5 + 0.5)

def dummy_linear(case):
    total_sum = sum(case)
    return total_sum/19 > 0.5  # 19 => maximum value among all upper borders
