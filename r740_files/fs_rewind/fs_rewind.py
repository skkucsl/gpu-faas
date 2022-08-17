'''
##### File system rewinder  #####
0. Will be modified
- Change below functions to some object's method
- Get hash key when object constructed
- Create lower links when obeject constructed

1. Function description
1-1. checkpointing
- Simply copy current 'diff' directory to 'chk' directory for restore
- Using shutil.copy2 (shutil.copytree's copying method) to keep metadata of files
1-2. restoration
- First, check 'diff' and 'chk' only -> find the modification
- If not exist files of 'chk' in 'diff', traverse the lower link
- If find at lower link (lower image's diff), restore to the nearest lower link's file
- If not found, remove the file

2. PATH description
2-1. Merged
- Mount point of current container
- Can modify at host system
2-2. Diff
- Different point compare to lower images
- Cannot modify and automatically updated by storage driver (maybe)
2-3. CHK
- Checkpointed directory which created by file system rewinder
2-4. Lower
- Links of current container's baseline images
- Divided by ':' and sorted
'''

import os
import shutil

ROOT_PATH = '/var/lib/docker/overlay2/'
HASH_KEY = 'c8971409d86218ad0f1159d5fcbaf082c60239f65d603fcf69d7857bb205fb68'
###### Will be moved into constructor ##########
MERGED_PATH = ROOT_PATH + HASH_KEY + '/merged/'
DIFF_PATH = ROOT_PATH + HASH_KEY + '/diff/'
CHK_PATH = ROOT_PATH + HASH_KEY + '/chk/'
LOWER_PATH = ROOT_PATH + HASH_KEY + '/lower'

REMOVAL = []
REMOVAL_DIR = []
REWIND = []
LOWER_LINK = []

with open(LOWER_PATH, 'r') as l:
    lower = l.readline()
    LOWER_LINK = lower.split(':')
#################################################

# Will be changed to real constructor
def constructor(hash_key):
    HASH_KEY = hash_key
    # Then each path will be changed like description above
    with open(LOWER_PATH, 'r') as l:
        lower = l.readline()
        LOWER_LINK = lower.split(':')


def change_path(path, cases):
    res = path
    if cases == 'd2c':
        res = path.replace(DIFF_PATH, CHK_PATH)
    elif cases == 'd2m':
        res = path.replace(DIFF_PATH, MERGED_PATH)
    elif cases == 'rm':
        res = path.replace(DIFF_PATH, '')
    return res

def is_modify(path):
    diff_stat = os.stat(path)
    chk_stat = os.stat(change_path(path, 'd2c'))
    if diff_stat.st_mtime == chk_stat.st_mtime:
        return False
    else:
        return True

def get_origin(path):
    # Traverse "lower" and find last modified version of "path"
    res = ''
    for links in LOWER_LINK:
        if os.path.exists(ROOT_PATH + links + '/' + change_path(path,'rm')):
            res = ROOT_PATH + links + '/' + change_path(path,'rm')
            break

    return res

def file_from_lower(path):
    # Check if file is in lower links, and restore if exist
    file_origin = get_origin(path)
    if file_origin == '':
        REMOVAL.append(path)
    else:
        # Restore from lower links (Cannot process at below code)
        shutil.copy2(file_origin, change_path(path, 'd2m'))

def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def cleaning():
    # Clean up each object which used in checkpointing() and restoration()
    shutil.rmtree(CHK_PATH)

def checkpointing():
    shutil.copytree(DIFF_PATH, CHK_PATH)

def restoration():
    for root, dirs, files in os.walk(DIFF_PATH):
        # Skip the removal directory
        if len(REMOVAL_DIR) > 0 and root.find(REMOVAL_DIR[-1]) == 0:
            continue

        # When directory exist on checkpoint
        if os.path.exists(change_path(root,'d2c')):
            for f in files:
                if os.path.exists(change_path(root+'/'+f,'d2c')):
                    # modification check and add into REWIND
                    if is_modify(root+'/'+f):
                        REWIND.append(root+'/'+f)
                else:
                    file_from_lower(root+'/'+f)
        # If not, find from lower links
        else:
            origin_path = get_origin(root)
            if origin_path == '':
                # This directory is created
                REMOVAL_DIR.append(root)
            else:
                for f in files:
                    file_from_lower(root+'/'+f)
                        

    # Remove
    REMOVAL.extend(REMOVAL_DIR)
    for path in REMOVAL:
        rm_path = change_path(path,'d2m')
        remove(rm_path)

    # Restore (only file)
    for path in REWIND:
        shutil.copy2(change_path(path,'d2c'), change_path(path,'d2m'))

    cleaning()

