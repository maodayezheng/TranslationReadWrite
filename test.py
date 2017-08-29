import sys
import numpy as np
import json
import theano.tensor as T
import theano

with open("SentenceData/BPE/news2013.tok.bpe.32000.txt", "r") as news:
    data = json.loads(news.read())

data = sorted(data, key=lambda d: max(len(d[0]), len(d[1])))

grouped = []
g = []
length = 10
for d in data:
    if 9 < len(d[1]) == length:
        g.append(d)
    elif len(d[1]) > length:
        print(len(g))
        print(length)
        print("")
        grouped.append(g)
        g = [d]
        length += 1
    if length > 50:
        break

with open("SentenceData/BPE/grouped_news2013.tok.bpe.32000.txt", "w") as sep:
    sep.write(json.dumps(grouped))


