import numpy as np
from numpy import array
import json
import string
import time
np.set_printoptions(threshold=1000000)

good = []
with open("SentenceData/dev_idx.txt", "r") as dataset:
    test_data = json.loads(dataset.read())
    for pair in test_data:
        if pair[2] <= 30:
            good.append(pair)

with open("SentenceData/dev_idx_small.txt", "w") as dataset:
    dataset.write(json.dumps(good))