import numpy as np
import operator
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

results = np.loadtxt('remove_ONet.txt')

# results = sorted(results, key=lambda x: (results[:][0], results[:][1]))
results = sorted(results, key=operator.itemgetter(0), reverse=True)
# print(results)
total_objects = 159424

df = pd.DataFrame(results, columns=['confidence', 'IoU', 'TP', 'FP'])
print(np.sum(results[0:][2]))
print(np.array(results)[:, 2])
print(np.sum(df['TP']))
print(np.sum(results[0:][3]))
print(np.sum(df['FP']))

total_count = 0
tp_count = 0
fp_count = 0
graph_list = []
for item in results:
    total_count += 1
    if item[2] == 1.:
        tp_count += 1


# plt.figure(figsize=(40, 5))
# # sns.lineplot(data=df, x=df['TP']/total_objects, y=df['TP']/(df['TP']+df['FP']))
# sns.lineplot(data=df, x=df['TP'], y=df['FP'])
# plt.title('PR Curve(ONet 제거)')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()
