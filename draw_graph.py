import numpy as np
import operator
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# results = np.loadtxt('origin.txt')
results = np.loadtxt('remove_ONet.txt')

# results = sorted(results, key=lambda x: (results[:][0], results[:][1]))
results = np.array(sorted(results, key=operator.itemgetter(0), reverse=True))
# print(results)
total_objects = 159424

item_count = 0
tp_count = 0
fp_count = 0
graph_list = []
for item in results:
    item_count += 1
    if item[2] == 1.:
        tp_count += 1
    else:
        fp_count += 1
    graph_list.append([item[0], item_count, tp_count, fp_count])

graph_df = pd.DataFrame(graph_list, columns=['confidence', 'item_count', 'TP_count', 'FP_count'])

plt.figure(figsize=(40, 6))
# sns.lineplot(data=df, x=df['TP']/total_objects, y=df['TP']/(df['TP']+df['FP']))
sns.lineplot(data=graph_df, x=graph_df['TP_count']/total_objects, y=graph_df['TP_count']/graph_df['item_count'])
sns.lineplot(x=graph_df['TP_count']/total_objects, y=np.sum(results[:, 2])/(np.sum(results[:, 2])+total_objects))
# plt.title('PR Curve(Original)')
plt.title('PR Curve(Remove ONet)')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Recall')
plt.yticks(np.arange(0.0, 1.2, 0.1))
plt.ylabel('Precision')
plt.show()
