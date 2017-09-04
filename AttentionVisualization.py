import seaborn as sns
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

sns.set()
address = np.load("Translations/show/Frac/10/2014/15_address.npy")
print(address.shape)
address = address[:, 14, :-1]
print(address.shape)

#address = DataFrame(address, index=idx)
attention = np.load("Translations/show/Frac/10/2014/15_attention.npy")
print(attention.shape)
attention = attention[:18, 14]
print(attention.shape)
#print(attention)
#print(address)
mask = np.equal(address, 0.0)
heat_map = sns.heatmap(np.transpose(address), linewidths=.5, cmap="YlGnBu", mask=np.transpose(mask))
#heat_map = sns.heatmap(attention, linewidths=.5, cmap="YlGnBu")
idx = "He broke any resistance by means of violence and the issuing of threats .".split(" ")
idx.reverse()
heat_map.set_yticklabels(idx, rotation=360)
heat_map.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
plt.savefig("Frac10")
plt.show()
