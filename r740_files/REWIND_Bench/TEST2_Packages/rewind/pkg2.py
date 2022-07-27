import time
import mypy
import django

def init_time(args):
    return {'startTime': time.time() }
