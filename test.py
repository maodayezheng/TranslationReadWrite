import sys
import numpy as np
import json
import theano.tensor as T
import theano

with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as dev:
    validation_data = json.loads(dev.read())
    chosen = []
    for v in validation_data:
        if 5 <= len(v[0]) <=50:
            chosen.append(v)
    print(len(chosen))