import json
# load vocab
en_vocab = []
with open("", "r") as en_v:
    for v in en_v:
        en_vocab.append(v.rstrip("\n"))

de_vocab = []
with open("", "r") as de_v:
    for v in de_v:
        de_vocab.append(v.rstrip("\n"))

# load idx
en_idx = None
with open("", "r") as en:
    en_idx = json.loads(en)

de_idx = None
with open("", "r") as de:
    de_idx = json.loads(de)

# write sentence
with open("", "w") as en:
    for s in en_idx:
        sentence = ""
        for idx in s:
            sentence += en_vocab[idx]
            sentence += " "
        en.write(sentence+"\n")


with open("", "w") as de:
    for s in de_idx:
        sentence = ""
        for idx in s:
            sentence += de_vocab[idx]
            sentence += " "
        de.write(sentence+"\n")