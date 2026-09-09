import numpy as np



x = np.array([10.0, 6.0, 3.0,
            4.5, 5.6, 4.3,
            6.7, 5.6, 4.5,
            4.5, 1.2, 3.2,
            5.0, 6.0, 7.0])

x1 = np.array([10.0, 6.0, 3.0,
            4.5, 5.6, 4.3,
            6.7, 5.6, 4.5,
            4.5, 1.2, 3.2,
            5.0, 6.0, 7.0])

np.random.seed(42)

W = np.array([x, x1])

print(W.shape)

b = np.random.randn(2)

z = W @ x + b

relu = np.maximum(0, z)


print("x =", x.shape)  
print("W = ", W.shape)  
print("b = ", b.shape)  
print("z = ", z.shape)


print("z = ", z)

assert x.shape == (15,)
assert W.shape == (2,15)
assert b.shape == (2,)
assert z.shape == (2,)

print("z =", z)
print("ReLU(z) =", relu)

assert z.shape == (2,)
assert relu.shape == (2,)