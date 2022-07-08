#!/usr/bin/python3

import time
import json

def main(args):
	startTime = time.time()
	a = 1
	for i in range(10000):
		a = a + 4
	return {'startTime': startTime}
