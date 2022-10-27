import sys, json
from numpy import matrix, linalg, random
import time
import os

def linpack(n):
    # LINPACK benchmarks
    ops = (2.0 * n) * n * n / 3.0 + (2.0 * n) * n

    # Create AxA array of random numbers -0.5 to 0.5
    A = random.random_sample((n, n)) - 0.5
    B = A.sum(axis=1)

    # Convert to matrices
    A = matrix(A)
    B = matrix(B.reshape((n, 1)))

    # Ax = B
    start = time.time()
    x = linalg.solve(A, B)
    end = time.time()
    
    latency = end - start
    mflops = (ops * 1e-6 / latency)

    result = {
        'mflops': mflops,
        'latency': latency
    }

    return result


def main(args):
    startTime = time.time()
    num = int(args.get('n', '1000'))
    result = linpack(num)
    endTime = time.time()
    result['startTime'] = startTime
    result['functionTime'] = endTime - startTime
    print(json.dumps(result))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = dict()
    main(args)
