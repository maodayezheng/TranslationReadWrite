import sys
import numpy as np
import json

with open("SentenceData/dev_idx_small.txt", "r") as dataset:
    test_data = json.loads(dataset.read())
mini_batch = test_data[:2000]
mini_batch = sorted(mini_batch, key=lambda d: d[2])
print(len(mini_batch))
data = []
for m in mini_batch:
    print(m)
    if m[2] > 17:
        data.append(m)
print(len(data))

with open("SentenceData/dev_1000.txt", "w") as dataset:
    dataset.write(json.dumps(data[:1000]))
