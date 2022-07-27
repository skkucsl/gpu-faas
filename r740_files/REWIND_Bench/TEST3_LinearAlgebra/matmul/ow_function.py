import numpy as np
import time


def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = np.matmul(A, B)
    latency = time.time() - start
    return {'latency': latency}


def init_time(args):
    startTime = time.time()
    num = int(args.get('n', '1000'))
    result = matmul(num)
    result['startTime'] = startTime
    return result
