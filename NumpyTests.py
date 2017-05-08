import numpy as np
from numpy import array
import json
import string
import time
np.set_printoptions(threshold=1000000)
with open("SentenceData/WMT/Data/data_idx.txt", "r") as dataset:
    train_data = json.loads(dataset.read())

    batch_indices = np.random.choice(len(train_data), 600, replace=False)
    mini_batch = [train_data[ind] for ind in batch_indices]
    mini_batch = sorted(mini_batch, key=lambda d: d[2])

    unique_target = []
    start = time.clock()
    for m in mini_batch:
        unique_target += m[1]
    unique_target = np.unique(unique_target)

    num_samples = 8000 - len(unique_target)
    candidate = np.arange(30004)
    candidate = np.delete(candidate, unique_target, None)
    samples = np.random.choice(a=candidate, size=num_samples, replace=False)
    samples = np.concatenate([unique_target, samples])
    print(time.clock() - start)
    print(samples.shape)