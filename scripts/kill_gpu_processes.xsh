#!/usr/bin/env xonsh
#
sudo fuser -v /dev/nvidia* &> temp.txt

for line in $(cat temp.txt).splitlines():
    words = line.split()
    if words[0] == 'dhruveshp':
        kill -9 @(words[1])