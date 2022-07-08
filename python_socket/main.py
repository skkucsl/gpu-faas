import ctypes
from numba import cuda


from socket import *
import threading
import time

s = time.time()
temp = cuda.device_array((1,1))
test = ctypes.CDLL("./libtest.so")
fname = '_Z9init_timev'
func = getattr(test,fname)
func.restype = ctypes.c_double
e = time.time()
print("Init GPU: ")
print(e-s)

def receive(sock):
    while True:
        recvData = sock.recv(1024).decode('utf-8')
        print("Received: ", recvData)
        if recvData == 'run':
            print(time.time())
            init_time = func()
            sendData = str(init_time)

port = 40509

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('127.0.0.1', port))

print("Connected")

receiver = threading.Thread(target=receive, args=(clientSock,))

receiver.start()

while True:
    time.sleep(1)
    pass
