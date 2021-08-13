import numpy as np
import operator

import matplotlib.pyplot as plt
import seaborn as sns

results = np.loadtxt('remove_ONet.txt')

# results = sorted(results, key=lambda x: (results[:][0], results[:][1]))
results = sorted(results, key=operator.itemgetter(0), reverse=True)
print(results)
total_objects = 159424

plt.figure(figsize=(40, 5))
sns.lineplot(data=results, x=results[2]/total_objects, y=results[2]/(results[2]+results[3]))
plt.title('PR Curve(ONet 제거)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
