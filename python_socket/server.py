from socket import *
import threading
import time

def send(sock):
    while True:
        sendData = input('')
        sock.send(sendData.encode('utf-8'))

def receive(sock):
    while True:
        recvData = sock.recv(1024)
        print("Received: ", recvData.decode('utf-8'))

port = 40509

serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(('', port))
serverSock.listen(1)

print("Wait...")

connectionSock, addr = serverSock.accept()

print("Connected")

sender = threading.Thread(target=send, args=(connectionSock,))
receiver = threading.Thread(target=receive, args=(connectionSock,))

sender.start()
receiver.start()

while True:
    time.sleep(1)
    pass



