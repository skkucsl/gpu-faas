'''
    Original code from ServerlessBench-TestCase7 Python-hello
    Modification: Start time unit (ms (int) -> time.time() original)
'''

import time
import sys, json

def main(args):
    """Main."""
    startTime = time.time()
    name = args.get('name', 'stranger')
    greeting = 'Hello ' + name + '!'

    tmp = {'greeting': greeting, 'startTime': startTime}
    print(json.dumps(tmp))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = dict()
    main(args)
