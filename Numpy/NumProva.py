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
"""
np.random.seed(seed=18)
random_array_4 = np.random.randint(10, size=(3, 5))
print(random_array_4) 

random_array_5 = np.random.randint(10, size=(3, 5))
print(random_array_5) 


import secrets
def secure_rng(min_value, max_value):
    return secrets.randbelow(max_value - min_value + 1) + min_value
"""


#VIEWING ARRAYS AND MATRICES
"""
x = np.unique(random_array_4)
print(x)
print(random_array_4[0])
print(random_array_4[1:])
print(random_array_4[:2])

random_array_5 = np.random.randint(10, size=(2,3,4,5))

print(random_array_5)
print(random_array_5[:,:,:,:3]) 
"""

#MANIPULATING ARRAYS    

"""
a1 = np.array([1, 2, 3])

ones = np.ones(3)

a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])

a3 = np.array([
    [[1, 2, 3], [4, 5,6],[7, 8, 9]],
    [[10, 11, 12], [13, 14,15], [16, 17, 18]]   
               ]) 

x = a1 + ones
#print(x) 
x2 = a1 - ones
#print(x2) 
x3 = a1 *  a2
#print(x3)
x3a = a2 / a1
#print(x3a)
x3b = a2 // a1
#print(x3b)
x3c = a2 ** a1
#print(x3c)
x3d = np.sqrt(a2)
#print(x3d)
x3e = a2 % a1
#print(x3e)
x3f = np.add(a2, 8)
#print(x3f)
x3g = a2 % 2
#print(x3g)
x3h = np.exp(a1)
#print(x3h)
x3i = np.log(a1)
#print(x3i)
"""

#AGGREGATION

"""
a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])

x = np.mean(a2)
print(x)

x2 = np.max(a2)
print(x2)

x3 = np.min(a2)
print(x3)

x4 = np.std(a2)
print(x4)

x5 = np.var(a2)
print(x5)
"""


"""
import time

aBig = np.random.random(100000000)

start_time_numpy = time.time()
aggNumPy = np.sum(aBig)
end_time_numpy = time.time()
numpy_time = end_time_numpy - start_time_numpy

print("Aggregazione NumPy:", aggNumPy)
print("Tempo di esecuzione NumPy (secondi):", numpy_time)

start_time_python = time.time()
aggPy = sum(aBig)
end_time_python = time.time()
python_time = end_time_python - start_time_python

print("Aggregazione Python:", aggPy)
print("Tempo di esecuzione Python (secondi):", python_time)

#standard deviation and variance
"""
"""
import matplotlib.pyplot as plt

high_var_array = np.array([1,100,200,300,4000,5000])
low_var_array = np.array([2,4,6,8,10])

x = np.var(high_var_array)
y = np.var(low_var_array)

print(x, y)

plt.hist(high_var_array)

print(plt.show())
"""

#reshape and transpose

"""
a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])
a3 = np.array([
    [[1, 2, 3], [4, 5,6],[7, 8, 9]],
    [[10, 11, 12], [13, 14,15], [16, 17, 18]]   
               ]) 

q = a2.reshape(2,3,1)
print(q)

x = q * a3
print(x)

w = a3.T
print(w)
"""

#Dot product vs Element-wise 
"""
a3 = np.array([
    [[1, 2, 3], [4, 5,6],[7, 8, 9]],
    [[10, 11, 12], [13, 14,15], [16, 17, 18]]   
               ]) 

np.random.seed(seed=0)

mat1 = np.random.randint(10, size=(5, 3))
mat2 = np.random.randint(10, size=(5, 3))
print(mat1 , mat2)

x =  mat1 * mat2
print(x)

#np.dot(mat1, mat2) dot product non si può usare in questo caso perchè abbiamo una colonna di troppo

mat3 = np.dot(mat1, mat2.T)
print(mat3)
"""

#ESERCIZIO
"""
import pandas as pd

np.random.seed(seed=0)
sales_amounts = np.random.randint(20, size=(5,3))

weekly_sales = pd.DataFrame(sales_amounts,
                            index=["Mon", "Tue", "Wed", "Thu", "Fri"],
                            columns=["Almond butter","Peanut butter","Cashew butter"]
                            )
                            
print(weekly_sales)

prices = np.array([10, 8, 12])

butter_prices = pd.DataFrame(prices.reshape(1, 3),
                             index=["Price"],
                             columns=["Almond butter","Peanut butter","Cashew butter"]
                            )

daily_sales = butter_prices.dot(weekly_sales.T)

weekly_sales["Total ($)"] = daily_sales.T

print(weekly_sales)
"""

#COMPARIOSON OPERATORS

"""
a1 = np.array([1, 2, 3])     

a2 = np.array([[1, 2.0, 3.3], [4, 5, 6.5]])

print(a1 < a2)
print(a2 >= 3)
print(a1 == a2)
"""

#SORTING ARRAYS

"""
random_array_5 = np.random.randint(10, size=(3,5))

x = np.sort(random_array_5)
print(x)

y = np.argsort(random_array_5)
print(y)

z = np.argmin(random_array_5)
print(z)

za = np.argmax(random_array_5, axis=1)
print(za)

"""

#IMAGE TO ARRAY
"""
from matplotlib.image import imread

serpente = imread("Java_PY.png")
print(serpente.size, serpente.shape, serpente.ndim)
"""
