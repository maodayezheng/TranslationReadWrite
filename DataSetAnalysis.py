import json
import operator

occurent_count = {}
vocab = []
with open("SentenceData/vocab_de", "r", encoding="utf8") as v:
    for line in v:
        vocab.append(line.strip("\n"))

count = 0
with open("SentenceData/data_idx_small.txt", "r") as dataset:
    train_data = json.loads(dataset.read())
    for d in train_data:
        count += 1
        sentence = d[1]
        idx = sentence[2]
        word = vocab[idx]
        if word in occurent_count:
            occurent_count[word] += 1
        else:
            occurent_count[word] = 1

occurent_count = sorted(occurent_count.items(), key=operator.itemgetter(1), reverse=True)
print(len(occurent_count))
print(count)
print(occurent_count)