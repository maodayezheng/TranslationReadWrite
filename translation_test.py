from ModelUtils.ParamFreeVAE.DeepReluIORNN.FourLayersV2 import DeepReluTransReadWrite
import pickle as cPickle
import json
import sys
import numpy as np

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)
main_dir = sys.argv[0]
out_dir = sys.argv[2]

restore_date = ""
restore_param = ""
test_file = ""
check_prediction = False

if __name__ == '__main__':
    print("Start testing translation experiment")
    print("Decoding the sequence")
    test_data = None
    model = DeepReluTransReadWrite()

    # Load vocabulary
    vocab = []
    with open("SentenceData/BPE/vocab.bpe.32000", "r", encoding="utf8") as v:
        for line in v:
            vocab.append(line.strip("\n"))

    with open("code_outputs/"+restore_date + restore_param, "rb") as params:
        model.set_param_values(cPickle.load(params))
    with open("SentenceData/BPE/" + test_file, "r") as dataset:
        test_data = json.loads(dataset.read())
    chosen = []

    decode = model.decode_fn()
    sour_sen = []
    refe_sen = []
    forc_sen = []
    gred_sen = []
    group_id = 10
    for m in test_data:
        l = max(len(m[-1, 0]), len(m[-1, 1]))
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
        force_max, prediction = decode(source, target)

        # Map the index to string
        with open("code_outputs/" + out_dir + "/source_" + str(group_id) + ".txt", "w") as sour, \
             open("code_outputs/" + out_dir + "/reference_" + str(group_id) + ".txt", "w") as refe, \
             open("code_outputs/" + out_dir + "/prediction_" + str(group_id) + ".txt", "w") as pred:

            for n in range(len(m)):
                s = source[n, 1:]
                t = target[n, 1:]
                f = force_max[:, n]
                p = prediction[:, n]

                sentence = ""
                for s_idx in s:
                    if s_idx == 1 or s_idx == -1:
                        break
                    sentence += (vocab[s_idx] + " ")
                if check_prediction:
                    print(sentence)

                sour.write(sentence + "\n")
                sentence = ""
                for t_idx in t:
                    if t_idx == 1 or t_idx == -1:
                        break
                    sentence += (vocab[t_idx] + " ")
                if check_prediction:
                    print(sentence)
                refe.write(sentence + "\n")
                sentence = ""
                for idx in p:
                    if idx == 1:
                        break
                    sentence += (vocab[idx] + " ")
                if check_prediction:
                    print(sentence)
                pred.write(sentence + "\n")

                if check_prediction:
                    for t_idx in f:
                        if t_idx == 1 or t_idx == -1:
                            break
                        sentence += (vocab[t_idx] + " ")
                    print(sentence)
        group_id += 1
