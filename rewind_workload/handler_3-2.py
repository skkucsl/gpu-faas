import numpy as np
import time


def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = np.matmul(A, B)

    end = time.time()
    latency = end - start
    return {'latency': latency}


def main(args):
    startTime = time.time()
    num = int(args.get('n', '1000'))
    result = matmul(num)
    endTime = time.time()
    result['startTime'] = startTime
    result['functionTime'] = endTime - startTime
    return result
