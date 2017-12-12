import numpy as np


def f(a, b=[]):
    b.append(a)
    print(len(b))


f(1)
f(1)
f(1)
f(1)
b = [2, 4, 5, 6]
for i in b:
    if not i % 2:
        b.remove(i)
print(b)

arr1 = np.ones([12,2])
print(arr1)
print(arr1.transpose())
arr = np.ones(12)
print(arr)
print(arr.transpose())
