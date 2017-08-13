import numpy as np
import matplotlib.pyplot as plt

folder = "code_outputs/"
date = "2017_08_09_14_52_53/"
loss = "validation_loss.npy"
valid_loss = np.load(folder+date+loss)
print(valid_loss[:-1].shape)
#attn_loss = np.mean(attn_loss[:-1].reshape((6, 25000)), axis=-1)

plt.plot(-valid_loss, label='valid_loss', zorder=3)
plt.xlabel('Negative log likelihood (Averaged over 1000 iterations)')
plt.ylabel('L(X))')
plt.legend(loc=2)
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