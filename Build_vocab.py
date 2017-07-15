import json
import numpy as np

np.set_printoptions(threshold=1000000)
en_vocab = {}
de_vocab = {}
min_len = 15
max_len = 50
sentence_pair = []
pair_count = 0
selected_en = []
selected_de = []
with open("SentenceData/translation/10sentenceTest/en.txt", "r") as en, \
     open("SentenceData/translation/10sentenceTest/de.txt", "r") as de:
    c = 0
    for e_line in en:
        c += 1
        d_line = de.readline()
        e_sentence = e_line.rstrip('\n').split(" ")
        d_sentence = d_line.rstrip('\n').split(" ")
        if 15 < len(e_sentence) <= 30 and 15 < len(d_sentence) <= 30:
            pair_count += 1
            selected_en.append(e_line)
            selected_de.append(d_line)
            for token in e_sentence:
                if token in en_vocab:
                    en_vocab[token] += 1
                else:
                    en_vocab[token] = 1

            for token in d_sentence:
                if token in de_vocab:
                    de_vocab[token] += 1
                else:
                    de_vocab[token] = 1
        if c == 3000:
            break

en_vocab = sorted(en_vocab.items(), key=lambda d: d[1], reverse=True)
de_vocab = sorted(de_vocab.items(), key=lambda d: d[1], reverse=True)


with open("SentenceData/translation/10sentenceTest/vocab_en", "w") as en:
    count = 0
    for e in en_vocab:
        count += 1
        en.write(e[0]+'\n')
        if count == 40000:
            break

with open("SentenceData/translation/10sentenceTest/vocab_de", "w") as de:
    count = 0
    for e in de_vocab:
        count += 1
        de.write(e[0]+'\n')
        if count == 40000:
            break
