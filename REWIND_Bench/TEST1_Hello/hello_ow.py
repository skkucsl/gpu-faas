'''
    Original code from ServerlessBench-TestCase7 Python-hello
    Modification: Start time unit (ms (int) -> time.time() original)
'''

import time
import json
def main(args):
    """Main."""
    startTime = time.time()
    name = args.get('name', 'stranger')
    greeting = 'Hello ' + name + '!'
    return {'greeting': greeting, 'startTime': startTime}
