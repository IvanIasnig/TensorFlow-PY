import numpy as np

a1 = np.array([1, 2, 3])     


a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])


a3 = np.array([
    [[1, 2, 3], [4, 5,6],[7, 8, 9]],
    [[10, 11, 12], [13, 14,15], [16, 17, 18]]   
               ]) 
print(a1)
print(a2)
print(a3)

print(a1.shape) # shape = (3,)
print(a2.shape) #shape = (2,3)
print(a3.shape) #shape= (2,3,3)

print(a1.ndim) #ndim = 1
print(a2.ndim) #ndim = 2
print(a3.ndim) #ndim = 3

print(a1.dtype) #dtype = int32
print(a2.dtype) #dtype = float
print(a3.dtype) #dtype = int32

print(a1.size) #size = 3
print(a2.size) #size = 6
print(a3.size) #size = 18

print(type(a1)) #<class 'numpy.ndarray'>
print(type(a2)) #<class 'numpy.ndarray'>
print(type(a3)) #<class 'numpy.ndarray'>

print(a1.itemsize) #itemsize = 4
print(a2.itemsize) #itemsize = 8
print(a3.itemsize) #itemsize = 4

print(a1.strides) #strides = (4,)
print(a2.strides) #strides = (8,8)
