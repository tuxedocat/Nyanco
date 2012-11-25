#! /usr/bin/env python
# coding: utf-8
'''
betterpool.py

A pool like class which can be used from functions inside classes
'''
from multiprocessing import Process, Pipe


class Pool2:
    def __init__(self, proc_num=8):
        self.proc_num = proc_num

    def map(self, func, args):
        def pipefunc(conn,arg):
            conn.send(func(arg))
            conn.close()
        retl = []
        k = 0
        while(k < len(args)):
            plist = []
            clist = []
            end = min(k + self.proc_num, len(args))
            for arg in args[k:end]:
                pconn, cconn = Pipe()
                plist.append(Process(target = pipefunc, args=(cconn,arg,)))
                clist.append(pconn)
            for p in plist:
                p.start()
            for conn in clist:
                retl.append(conn.recv())
            for p in plist:
                p.join()
            k += self.proc_num
        return retl