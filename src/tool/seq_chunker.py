#! /usr/bin/env python
# coding: utf-8
'''
seq_chunker.py
'''

chunk_gen = lambda x,y: (x[i:i+y] for i in range(0,len(x),y))