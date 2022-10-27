import time
import random
import string
import pyaes
import os

def generate(length):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def main(args):
    startTime = time.time()

    length_of_message = int(args.get('leng', '1024'))
    num_of_iterations = int(args.get('num', '10'))

    message = generate(length_of_message)

    # 128-bit key (16 bytes)
    KEY = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

    start = time.time()
    for loops in range(num_of_iterations):
        aes = pyaes.AESModeOfOperationCTR(KEY)
        ciphertext = aes.encrypt(message)

        aes = pyaes.AESModeOfOperationCTR(KEY)
        plaintext = aes.decrypt(ciphertext)
        aes = None
    
    endTime = time.time()
    latency = endTime - start

    return {'latency': latency, 'startTime': startTime, 'functionTime': endTime - startTime}

