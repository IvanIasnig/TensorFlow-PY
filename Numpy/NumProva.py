import numpy as np
"""
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
"""

#CREATING ARRAYS
"""
sample_array = np.array([1, 2, 3])

pippo = np.ones(sample_array, dtype=np.int64) #dtype facoltativo
print(pippo) #array([1, 1, 1])

pluto = np.zeros((4,5), dtype=np.int64)
print(pluto) #[[0 0 0 0 0][0 0 0 0 0][0 0 0 0 0][0 0 0 0 0]]

range_array = np.arange(10, 23, 4)
print(range_array) #array([10, 14, 18, 22])

random_array = np.random.randint(0, 10, size=(3, 5))
print(random_array)

random_array2 = np.random.random(size=(3, 5))
print(random_array2)

random_array3 = np.random.rand(5,3)
print(random_array3)
"""

#RANDOM SEED

np.random.seed(seed=42)
random_array_4 = np.random.randint(10, size=(3, 5))
print(random_array_4) 

#random_array_5 = np.random.randint(10, size=(3, 5))
#print(random_array_5) 

#randomizzatore sicuro
#import secrets
#def secure_rng(min_value, max_value):
#    return secrets.randbelow(max_value - min_value + 1) + min_value



#VIEWING ARRAYS AND MATRICES

x = np.unique(random_array_4)
print(x)
print(random_array_4[0])
print(random_array_4[1:])
print(random_array_4[:2])

random_array_5 = np.random.randint(10, size=(2,3,4,5))

print(random_array_5)
print(random_array_5[:,:,:,:3]) 