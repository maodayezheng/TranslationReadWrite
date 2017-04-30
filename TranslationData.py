import json
import numpy as np

np.set_printoptions(threshold=1000000)
max_len = 50
# Read the English Vocab
english_vocab = {}
en_idx = 0
with open("SentenceData/WMT-raw/vocab.50K.en", "r") as en:
    for line in en:
        english_vocab[line.rstrip('\n')] = en_idx
        en_idx += 1

# Read the German Vocab
german_vocab = {}
de_idx = 0
with open("SentenceData/WMT-raw/vocab.50K.de", "r") as de:
    for line in de:
        german_vocab[line.rstrip('\n')] = de_idx
        de_idx += 1

data_pair = []
count = 0
with open("SentenceData/WMT-raw/train.en", "r") as en:
    for line in en:
        count += 1

en_total = 0
en_occur = [0] * en_idx
de_total = 0
de_occur = [0] * de_idx
pair_count = 0
with open("SentenceData/WMT-raw/train.en", "r") as en, open("SentenceData/WMT-raw/train.de", "r") as de:
    for en_line in en:
        sentence = en_line.rstrip('\n')
        en_tokens = sentence.split(" ")
        de_line = de.readline()
        sentence = de_line.rstrip('\n')
        de_tokens = sentence.split(" ")
        if 15 < len(de_tokens) <= max_len and 15 < len(en_tokens) <= max_len:
            l = []
            pair_count += 1
            en_idx = []
            for e_t in en_tokens:
                en_total += 1
                e_i = english_vocab.get(e_t, 3)
                en_idx.append(e_i)
                en_occur[e_i] += 1
            l.append(len(en_idx))
            if len(en_idx) < max_len:
                en_idx = [2] + en_idx + [1] + [0] * (max_len - len(en_idx))
            else:
                en_idx = [2] + en_idx + [1]

            de_idx = []
            for d_t in de_tokens:
                de_total += 1
                d_i = german_vocab.get(d_t, 3)
                de_idx.append(d_i)
                de_occur[d_i] += 1
            l.append(len(de_idx))
            if len(de_idx) < max_len:
                de_idx = [2] + de_idx + [1] + [0] * (max_len - len(de_idx))
            else:
                de_idx = [2] + de_idx + [1]
            data_pair.append([en_idx, de_idx, l])

# Sample a fixed set for each word based on word occurrence

en_unigram = np.array(en_occur)/en_total
de_unigram = np.array(de_occur)/de_total
num_sample = 300
en_c_id = np.arange(len(english_vocab))
de_c_id = np.arange(len(german_vocab))
en_candidates = []
de_candidates = []

# de
for i in range(len(german_vocab)):
    # Get the german samples
    d_gram = de_unigram
    temp = d_gram[i]
    d_gram[i] = 0
    d_norm = np.sum(d_gram)
    unigram = d_gram/d_norm
    de_sample = np.append(np.random.choice(size=(num_sample,), a=de_c_id, p=unigram, replace=False), [i])
    de_candidates.append(de_sample.tolist())
    d_gram[i] = temp

with open("SentenceData/WMT/Data/data_idx.txt", "w") as dataset:
    d = json.dumps(data_pair)
    dataset.write(d)

with open("SentenceData/WMT/Data/de_candidate_sample.txt", "w") as s:
    d = json.dumps(de_candidates)
    s.write(d)