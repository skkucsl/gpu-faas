import sys, json
import time
import mypy
import django

def main(args):
    tmp =  {'startTime': time.time() }
    print(json.dumps(tmp))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = json.loads(sys.argv[1])
    else:
        args = dict()
    main(args)
