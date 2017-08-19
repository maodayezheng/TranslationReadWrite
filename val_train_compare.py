import numpy as np
import matplotlib.pyplot as plt

folder = "Translations/show/loss/"
io_loss = np.load(folder + "io_validation_loss.npy")
ios_att_loss = np.load(folder + "ios_att_validation_loss.npy")
ios_loss = np.load(folder + "ios_validation_loss.npy")
seq2seq_loss = np.load(folder + "seq2seq_validation_loss.npy")
seq2seq_att_loss = np.load(folder + "seq2seq_att_validation_loss.npy")
simple_loss = np.load(folder + "simple_validation_loss.npy")
simple_att_loss = np.load(folder + "simple_att_validation_loss.npy")


plt.plot(-io_loss, label='io', zorder=1)
#plt.plot(-ios_att_loss, label='ios_att', zorder=2)
plt.plot(-ios_loss, label='ios', zorder=3)
plt.plot(-seq2seq_loss, label='seq2seq', zorder=4)
#plt.plot(-seq2seq_att_loss, label='seq2seq_att', zorder=5)
plt.plot(-simple_loss, label='simple', zorder=6)
#plt.plot(-simple_att_loss, label='simple_att', zorder=7)


plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()
plt.show()


"""
total_elbos =0
for i in range(len(training)):
    total_elbos += training[i]
    if (i+1) % 200 is 0:
        aver_elbos.append(total_elbos/200)
        total_elbos = 0

plt.plot(aver_elbos, label='train', zorder=2)
plt.plot(validation, label='test', zorder=1)

plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()

plt.savefig('code_outputs/DecoderAddress-10k' + '.png')

plt.clf()
"""