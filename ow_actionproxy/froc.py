import ctypes
from numba import cuda


from socket import *
import threading
import time

lib_name = "/action/libow.so"
fname = 'init_time'

def receive(sock):
    while True:
        recvData = sock.recv(1024).decode('utf-8')
        print("Received: ", recvData)
        if recvData == 'run':
            init_time = func()
            sendData = str(init_time)

if (os.path.isfile(lib_name) and os.access(lib_name, os.R_OK)):
    stream = os.popen("nm -D " + self.lib + " | grep " + self.func_name + " | tail -n 1 | awk '{ print $3 }'")
    output = stream.read()
    output = output.replace('\n','')
    temp = cuda.device_array((1,1))
    FUNC_DL = ctypes.CDLL(self.lib)
    func = getattr(FUNC_DL, output)
    func.restype = ctypes.c_double

port = 40509

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('127.0.0.1', port))

#print("Connected")

receiver = threading.Thread(target=receive, args=(clientSock,))

receiver.start()

while True:
    time.sleep(1)
    pass
