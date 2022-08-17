#import ctypes
#from numba import cuda

from socket import *
import threading
import time
import os
import json

lib_name = "/action/ow_function.py"
f = 'init_time'
func = None

'''
def debug(msg):
    test_file = open('/root/ow_test/test.txt','a')
    test_file.write(msg)
    test_file.close()
'''

def init_func(fname):
    #debug("Into the init_func()...\n")
    global func
    if (os.path.isfile(lib_name) and os.access(lib_name, os.R_OK)):
        # Function lib init
        import ow_function as func
        return "success"
    else:
        return "err"

def receive(sock):
    while True:
        recvData = sock.recv(1024).decode('utf-8')
        if recvData == 'init':
            err = init_func(f)
            sock.send(err.encode('utf-8'))
        elif recvData == 'run':
            param = sock.recv(1024).decode('utf-8')
            init_time = func.init_time(json.loads(param))
            sendData = json.dumps(init_time)
            sock.send(sendData.encode('utf-8'))

port = 40509

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('127.0.0.1', port))

receiver = threading.Thread(target=receive, args=(clientSock,))

receiver.start()

while True:
    time.sleep(1)
    pass
