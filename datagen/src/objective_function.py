import math
import time


# file where objective function is declared (dummy test)
def dummy(case):
    time.sleep(10)
    return round(math.sin(sum(case)) * 0.5 + 0.5)


def dummy_linear(case):
    total_sum = sum(case)
    return total_sum/19 > 0.5  # 19 => maximum value among all upper borders
