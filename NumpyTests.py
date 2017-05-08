import numpy as np
from numpy import array
import json
import string
import time
np.set_printoptions(threshold=1000000)
with open("SentenceData/WMT/Data/data_idx.txt", "r") as dataset:
    train_data = json.loads(dataset.read())

    batch_indices = np.random.choice(len(train_data), 300, replace=False)
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
    mini_batch = np.array(mini_batch)
    print(time.clock() - start)
    mini_batchs = np.split(mini_batch, 10)

    for m in mini_batchs:
        l = m[-1, -1]
        source = None
        target = None
        for datapoint in m:
            s = np.array(datapoint[0])
            t = np.array(datapoint[1])
            if len(s) != l:
                s = np.append(s, [-1] * (l - len(s)))
            if len(t) != l:
                t = np.append(t, [-1] * (l - len(t)))
            if source is None:
                source = s.reshape((1, s.shape[0]))
            else:
                source = np.concatenate([source, s.reshape((1, s.shape[0]))])
            if target is None:
                target = s.reshape((1, t.shape[0]))
            else:
                target = np.concatenate([target, t.reshape((1, t.shape[0]))])
