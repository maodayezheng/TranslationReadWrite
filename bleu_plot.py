import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


io4lv2_att = [np.nan, 20.66, 21.70, 20.25, 19.82, 19.74, 19.85, 17.81, 18.37, np.nan]
frac_07_15 = [np.nan, 19.33, 20.01, 19.40, 18.34, 18.20, 18.84, 16.91, 18.18, np.nan]
frac_05_15 = [np.nan, 17.60, 18.10, 17.85, 17.24, 17.10, 16.50, 15.90, 16.10, np.nan]
frac_03_15 = [np.nan, 15.46, 15.27, 15.23, 13.38, 14.11, 14.41, 13.50, 12.79, np.nan]
seq2seq_att =[np.nan, 19.03, 21.01, 20.20, 19.84, 19.10, 19.44, 17.01, 18.08, np.nan]
t =[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
sns.tsplot(io4lv2_att, time=t, condition="RNNSearchCE 1.0", color="blue")
sns.tsplot(frac_07_15, time=t, condition="RNNSearchCE 0.7", color="red")
sns.tsplot(frac_05_15, time=t, condition="RNNSearchCE 0.5", color="orange")
sns.tsplot(frac_03_15, time=t, condition="RNNSearchCE 0.3", color="green")
sns.tsplot(seq2seq_att, time=t, condition="RNNSearch", color="yellow")

plt.savefig("bleu_compare")
plt.show()
