import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

results = np.load('result.txt')

results = sorted(results, key=lambda x: (-results[0], -results[1]))
total_objects = 50000

plt.figure(figsize=(40, 5))
sns.lineplot(data=results, x=results[:][2]/total_objects, y=results[:][2]/(results[:][2]+results[:][3]))
plt.title('PR Curve(ONet 제거)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()