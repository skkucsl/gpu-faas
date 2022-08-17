import os

PATH = "/var/lib/docker/overlay2/cf6bdf97c124f90d16fa800547fb3ab4327934f379eda5d850ae6938bcbd3acf/lower"
ROOT = "/var/lib/docker/overlay2/"

f = open(PATH, 'r')
lower = f.readline()
lower_link = lower.split(':')

for links in lower_link:
    print(ROOT+links)
    print(os.listdir(ROOT+links))
    print("---------------------------")
