import numpy as np

import matplotlib.pyplot as plt

results = np.load('result.txt')

results = sorted(results, key=lambda x: (-results[0], -results[1]))

