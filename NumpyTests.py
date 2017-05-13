import numpy as np
from numpy import array
import json
import string
import time

small = []
with open("SentenceData/WMT/Data/data_idx.txt", "r") as dataset:
    train_data = json.loads(dataset.read())
    for d in train_data:
        if len(d) <= 30:
            small.append(d)

with open("SentenceData/WMT/Data/data_idx_small.txt", "w") as dataset:
    dataset.write(json.dumps(small))