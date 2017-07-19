import json
import numpy as np

np.set_printoptions(threshold=1000000)
max_len = 30
# Read the English Vocab
vocab = {}
en_idx = 0
with open("SentenceData/wmt16_en_de/vocab.bpe.32000", "r", encoding="utf8") as en:
    for line in en:
        vocab[line.rstrip('\n')] = en_idx
        en_idx += 1

data_pair = []
selected_en = []
selected_de = []
en_total = 0
en_occur = [0] * en_idx
de_total = 0
pair_count = 0
with open("SentenceData/wmt16_en_de/newstest2013.tok.bpe.32000.en", "r", encoding="utf8") as en, \
     open("SentenceData/wmt16_en_de/newstest2013.tok.bpe.32000.de", "r", encoding="utf8") as de:
    for en_line in en:
        sentence = en_line.rstrip('\n')
        en_tokens = sentence.split(" ")
        de_line = de.readline()
        sentence = de_line.rstrip('\n')
        de_tokens = sentence.split(" ")
        if 0 < len(de_tokens) <= max_len and 0 < len(en_tokens) <= max_len:
            selected_de.append(de_line)
            selected_en.append(en_line)
            l = []
            pair_count += 1
            en_idx = []
            for e_t in en_tokens:
                en_total += 1
                e_i = vocab[e_t]
                en_idx.append(e_i)
                en_occur[e_i] += 1

            de_idx = []
            for d_t in de_tokens:
                de_total += 1
                d_i = vocab[d_t]
                de_idx.append(d_i)

            de_idx = [0] + de_idx + [1]
            data_pair.append([en_idx, de_idx])


with open("SentenceData/newstest2013.tok.bpe.32000.txt", "w") as dataset:
    d = json.dumps(data_pair)
    dataset.write(d)

with open("SentenceData/newstest2013.tok.bpe.32000_en.txt", "w") as en:
    for line in selected_en:
        en.write(line + "\n")


with open("SentenceData/newstest2013.tok.bpe.32000_de.txt", "w") as de:
    for line in selected_de:
        de.write(line + "\n")

