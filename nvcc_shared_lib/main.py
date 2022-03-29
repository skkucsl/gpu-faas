import ctypes
from numba import cuda

temp = cuda.device_array((2,3))
test = ctypes.CDLL("./libtest.so")
test._Z9wait_timei(5)
test._Z9wait_timei(5)
