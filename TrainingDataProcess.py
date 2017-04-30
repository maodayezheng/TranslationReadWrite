import json
import numpy as np

np.set_printoptions(threshold=1000000)
en_vocab = {}
de_vocab = {}
min_len = 15
max_len = 50
sentence_pair = []
pair_count = 0
with open("SentenceData/WMT-raw/train.en", "r") as en, open("SentenceData/WMT-raw/train.de", "r") as de:
    for e_line in en:
        d_line = de.readline()
        e_sentence = e_line.rstrip('\n').split(" ")
        d_sentence = d_line.rstrip('\n').split(" ")
        if 15 < len(e_sentence) <= 50 and 15 < len(d_sentence) <= 50:
            pair_count += 1
            sentence_pair.append([e_sentence, d_sentence, [len(e_sentence), len(d_sentence)]])
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

en_vocab = sorted(en_vocab.items(), key=lambda d: d[1], reverse=True)
de_vocab = sorted(de_vocab.items(), key=lambda d: d[1], reverse=True)


with open("SentenceData/WMT/10000data-test/vocab_en", "w") as en:
    en.write("<PAD>" + "\n")
    en.write("<EOS>" + "\n")
    en.write("<UNK>" + "\n")
    count = 0
    for e in en_vocab:
        count += 1
        en.write(e[0]+'\n')
        if count == 20000:
            break

with open("SentenceData/WMT/10000data-test/vocab_de", "w") as de:
    de.write("<PAD>" + "\n")
    de.write("<EOS>" + "\n")
    de.write("<UNK>" + "\n")
    count = 0
    for e in de_vocab:
        count += 1
        de.write(e[0]+'\n')
        if count == 20000:
            break
