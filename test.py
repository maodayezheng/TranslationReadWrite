import sys
import numpy as np
import json
import theano.tensor as T
import theano


with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as dev:
     validation_data = json.loads(dev.read())
     validation_data = sorted(validation_data, key=lambda d: len(d[0]))

groups = []
l = 10
g = []
for v in validation_data:
    if len(v[0]) > l:
        print(len(g))
        groups.append(g)
        g =[]
        l += 1
    elif l > 50:
        break
    elif len(v[0]) > 9:
        g.append(v)

with open("SentenceData/BPE/grouped_news2013.tok.bpe.32000.txt", "w") as dev:
    group_data = json.dumps(groups)
    dev.write(group_data)
