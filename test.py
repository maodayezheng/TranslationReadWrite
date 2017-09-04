import sys
import numpy as np
import json
import theano.tensor as T
import theano

training_set = None
with open("SentenceData/BPE/news2014.tok.bpe.32000.txt", "r") as news:
    training_set = json.loads(news.read())

with open("SentenceData/BPE/news2015.tok.bpe.32000.txt", "r") as news:
    data = json.loads(news.read())
    for d in data:
        training_set.append(d)

data = sorted(training_set, key=lambda d: max(len(d[0]), len(d[1])))

grouped = []
g = []
length = 10
for d in data:
    if 9 < len(d[1]) <= length:
        g.append(d)
    elif len(d[1]) > length:
        print(len(g))
        print(length)
        print("")
        grouped.append(g)
        g = [d]
        length += 5
    if length > 50:
        break

with open("SentenceData/BPE/grouped_news.tok.bpe.32000.txt", "w") as sep:
    sep.write(json.dumps(grouped))


