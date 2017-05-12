import json

subset = []
with open("SentenceData/en-fr/data_idx_en_fr.txt", "r") as dataset:
    train_data = json.loads(dataset.read())

    for d in train_data:
        if d[2] <= 30:
            subset.append(d)

with open("SentenceData/en-fr/data_idx_small.txt", "w") as dataset:
      dataset.write(json.dumps(subset))