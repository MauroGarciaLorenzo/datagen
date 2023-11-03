import random
import time


# file where objective function is declared (dummy test)
def dummy(case):
    # time.sleep(10)
    return random.randint(0, 1)


def dummy_linear(case):
    total_sum = sum(case)
    return total_sum/19 > 0.5  # 19 => maximum value among all upper borders
