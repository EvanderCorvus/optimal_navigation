import numpy as np

x = np.array([1, 2, 3])
X, Y = np.meshgrid(x, x)
a = np.array(np.array(np.meshgrid(x, x)).T.reshape(-1, 2))


def f(x,y):
    return x+y**2
b = np.array([f(x,y) for x,y in a])
print(X)
print(Y)
print(b.reshape(X.shape).T)
