import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set()
attention = np.load("code_outputs/2017_08_29_16_03_55/20_address.npy")
attention = attention[1:19, -1, :-2]
mask = np.equal(attention, 0.0)
sns.heatmap(attention, linewidths=.5, cmap="YlGnBu", mask=mask)
plt.show()