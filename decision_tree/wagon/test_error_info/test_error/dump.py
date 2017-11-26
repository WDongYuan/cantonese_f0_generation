import numpy as np
a = np.loadtxt("/Users/weidong/Desktop/test_error/dump",delimiter=",")
print(a)
print(np.mean(a))
print(np.std(a))