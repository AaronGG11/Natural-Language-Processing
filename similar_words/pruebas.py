import numpy as np

a = [1,2,3,4,5]
b = [1,2,3,4,5]

ac = np.array(a)
bc = np.array(b)

print(list(np.multiply(ac,bc)))
print(type(bc))