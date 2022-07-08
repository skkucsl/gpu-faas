import os
import subprocess
import time

time.sleep(1)
p = subprocess.Popen(['/home/user01/gpu-faas/REWIND/microbench/exec'])
print(time.time())
(o, e) = p.communicate()
