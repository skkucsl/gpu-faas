import ctypes
from numba import cuda


from socket import *
import threading
import time
import os

lib_name = "/action/libow.so"
f = 'init_time'
func = None

def debug(msg):
    test_file = open('/root/ow_test/test.txt','a')
    test_file.write(msg)
    test_file.close()


def init_func(fname):
    #debug("Into the init_func()...\n")
    global func
    if (os.path.isfile(lib_name) and os.access(lib_name, os.R_OK)):
        # Find real function name in shared object
        #debug("Locate correctly\n")
        stream = os.popen("nm -D " + lib_name + " | grep " + fname + " | tail -n 1 | awk '{ print $3 }'")
        output = stream.read()
        output = output.replace('\n','')
        #debug("Function name get\n")
        # GPU init
        temp = cuda.device_array((1,1))
        #debug("GPU init done\n")
        # Function lib init
        FUNC_DL = ctypes.CDLL(lib_name)
        func = getattr(FUNC_DL, output)
        func.restype = ctypes.c_double
        #debug("Function lib load done\n")
        return "success"
    else:
        #debug("Error\n")
        return "err"

def receive(sock):
    while True:
        recvData = sock.recv(1024).decode('utf-8')
        #print("Received: ", recvData)
        if recvData == 'init':
            err = init_func(f)
            sock.send(err.encode('utf-8'))
            #debug("send init result\n")
        elif recvData == 'run':
            #debug("run\n")
            init_time = func()
            sendData = str(init_time)
            sock.send(sendData.encode('utf-8'))
            #debug("send run result\n")

port = 40509

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('127.0.0.1', port))

#print("Connected")

receiver = threading.Thread(target=receive, args=(clientSock,))

receiver.start()

while True:
    time.sleep(1)
    pass
