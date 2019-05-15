'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import sys

class Logger(object):
    '''
    example usage:

        stdout = Logger('log.txt')
        sys.stdout = stdout

        ... your code here ...

        stdout.delink()

    '''
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        bufsize = 0
        self.log = open(filename, "w", )
        
    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass 
