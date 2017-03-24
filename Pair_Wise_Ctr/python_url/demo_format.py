#!/usr/bin/env python
# -*- coding: gb18030 -*-

# Pw @ 2017-02-26 13:46:09
# @author XXX

import sys

print "<html>"
print "<body width=\"100%\">"
print "<table border=\"1\" width=\"600px\">"

for line in sys.stdin:
    line=line.strip()
    cache=line.split('\t')
    att=""
    if len(cache)>1:
        '''
        for item in cache[2:]:
            att=att+item.strip()+","
        '''
        #att=att+cache[2][:-2]+cache[3]
        att=cache[1]
        head="<tr><td><img src=\""+cache[0]+"\" height=\"200\" width=\"200\" /></td>"
        body="<td style=\"word-wrap;break-word\">"+att+"</td></tr>"
        print "%s%s" %(head,body)


print "</table>"
print "</body>"
print "</html>"



            
