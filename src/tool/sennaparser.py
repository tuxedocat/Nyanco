#!/usr/bin/env python
# coding: utf-8
import subprocess
import os


class SennaParser(object):
    def __init__(self, bin):
        assert isinstance(bin, unicode)
        subproc_args = { 'stdin': subprocess.PIPE,
                         'stdout': subprocess.PIPE,
                         'stderr': subprocess.STDOUT,  # not subprocess.PIPE
                         'cwd': '.',
                         'close_fds' : True}
        args = [bin + 'senna', '-path', bin]
        try:
            self.p = subprocess.Popen(args, **subproc_args)
        except OSError:
            raise 
        (self.stdouterr, self.stdin) = (self.p.stdout, self.p.stdin)

    def __del__(self):
        self.p.stdin.close() #send EOF (This is not obligate)
        try:
            self.p.kill()
            self.p.wait()
        except OSError:
            # can't kill a dead proc
            pass


    # def parseSentence(self, sentence):
        # self.p.stdin.write(sentence + "\n")
        # result = []
        # while True:
            # line = self.stdouterr.readline()[:-1]
            # if not line:
                # break
            # result.append(line.split())
        # return result


    def parseSentence(self, sentence):
        self.p.stdin.write(sentence + "\n")
        result = []
        while True:
            line = self.stdouterr.readline()[:-1]
            if not line:
                break
            result.append(line.split())
        self.stdouterr.flush()
        return result



if __name__ == '__main__':
    import sys 
    sennapath = unicode(os.environ["SENNAPATH"]) + u"/"
    senna = SennaParser(sennapath)
    while True:
        string = raw_input('input: ')
        if len(string) == 0:
            sys.exit()
        out = senna.parseSentence(string)
        print ">" , out


