from ModelUtils.ParamFreeVAE.DeepReluIORNN.FourLayersInterAttV2 import DeepReluTransReadWrite as TranslationModel
import pickle as cPickle
import json
import sys
import numpy as np
import os

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[0]
out_dir = sys.argv[2]

restore_param = "Translations/show/Param/io4lv2_att_final_model_params.save"
test_file = "grouped_news2015.tok.bpe.32000.txt"
check_prediction = False

if __name__ == '__main__':
    print("Start testing io version 4")
    test_data = None
    model = TranslationModel()

    # Load vocabulary
    vocab = []
    with open("SentenceData/BPE/vocab.bpe.32000", "r", encoding="utf8") as v:
        for line in v:
            vocab.append(line.strip("\n"))

    with open(restore_param, "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/BPE/" + test_file, "r") as dataset:
        test_data = json.loads(dataset.read())

    decode = model.decode_fn()
    sour_sen = []
    refe_sen = []
    forc_sen = []
    pred_sen = []
    test_data = np.array(test_data)
    for m in test_data[6:11]:
        m = sorted(m, key=lambda d: max(len(d[0]), len(d[1])))
        last_data = m[-1]
        l = max(len(last_data[0]), len(last_data[1]))
        source = None
        target = None
        # Pad the input sequence
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
                target = t.reshape((1, t.shape[0]))
            else:
                target = np.concatenate([target, t.reshape((1, t.shape[0]))])

        # Prediction
        [prediction, address, att_score] = decode(source, target)
        np.save(out_dir +"/" + str(l) + '_address.npy', address)
        np.save(out_dir + "/" + str(l) + '_attention.npy', att_score)
        # Map the index to string
        with open(out_dir + "/source_" + str(l) + ".txt", "w") as sour, \
             open(out_dir + "/reference_" + str(l) + ".txt", "w") as refe, \
             open(out_dir + "/prediction_" + str(l) + ".txt", "w") as pred:

            for n in range(len(m)):
                s = source[n]
                t = target[n, 1:]
                p = prediction[:, n]

                sentence = ""
                for s_idx in s:
                    if s_idx == 1 or s_idx == -1:
                        break
                    sentence += (vocab[s_idx] + " ")
                sour_sen.append(sentence)
                if check_prediction:
                    print(sentence)

                sour.write(sentence + "\n")
                sentence = ""
                for t_idx in t:
                    if t_idx == 1 or t_idx == -1:
                        break
                    sentence += (vocab[t_idx] + " ")
                refe_sen.append(sentence)
                if check_prediction:
                    print(sentence)
                refe.write(sentence + "\n")
                sentence = ""
                for idx in p:
                    if idx == 1:
                        break
                    sentence += (vocab[idx] + " ")
                pred_sen.append(sentence)
                if check_prediction:
                    print(sentence)
                pred.write(sentence + "\n")
    with open(out_dir + "/source.txt", "w") as sour:
        for s in sour_sen:
            sour.write(s + "\n")
    with open(out_dir + "/reference.txt", "w") as refe:
        for s in refe_sen:
            refe.write(s + "\n")
    with open(out_dir + "/prediction.txt", "w") as pred:
        for s in pred_sen:
            pred.write(s + "\n")
