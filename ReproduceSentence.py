import json
# load vocab
en_vocab = []
with open("SentenceData/vocab_en", "r", encoding="utf8") as en_v:
    for v in en_v:
        en_vocab.append(v.rstrip("\n"))

de_vocab = []
with open("SentenceData/vocab_de", "r", encoding="utf8") as de_v:
    for v in de_v:
        de_vocab.append(v.rstrip("\n"))

# load idx
idx = None
with open("SentenceData/dev_idx.txt", "r") as en:
    idx = json.loads(en.read())

# write sentence
with open("SentenceData/dev_en.txt", "w", encoding="utf8") as en, open("SentenceData/dev_de.txt", "w", encoding="utf8") as de:
    for s in idx:
        sentence = ""
        for i in s[0]:
            if i == 0 or i == 1:
                continue
            sentence += en_vocab[i]
            sentence += " "
        en.write(sentence+"\n")
        sentence = ""
        for i in s[1]:
            if i == 0 or i == 1:
                continue
            sentence += de_vocab[i]
            sentence += " "
        de.write(sentence + "\n")
