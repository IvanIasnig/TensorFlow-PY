import tensorflow as tf
"""
scalar = tf.constant(7)
print(scalar)

vector = tf.constant([10,10])
print(vector)

matrix = tf.constant([[10,7],
                      [7,10]])
print(matrix)

another_matrix = tf.constant([[10.,7.]
                              [3.,2.]
                              [8.,9.]], dtype=tf.float16 )
print(another_matrix)

tensor = tf.constant([[[1,2,3,],
                       [4,5,6]],
                       [[7,8,9],
                        [10,11,12]],
                        [[13,14,15],
                         [16,17,18]]])
    #A scalar is a single number
    #Vector is a number with direction (es: wind speed and direction)
    #Matrix is a 2-dimensional array o f numbers
    #Tensor is an n-dimensional array (so scalar, vetor and matrix are all tensors)
"""
#Creating tensors
"""
changable_tensor = tf.Variable([10,7])
unchangable_tensor = tf.constant([10,7])
print(changable_tensor,unchangable_tensor)

changable_tensor[0].assign(7)
print(changable_tensor)

#unchangable_tensor[0].assign(7) -> obviously it doesen't work 
"""
#Random tensors
"""
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
print(random_1)

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.uniform(shape=(3,2))
print(random_2) #nel video mi mostrava che se creavo random_2 uguale a raondom_1 erano uguali, il che è ovvio dato che hanno lo stesso seed, per provare al secondo gli ho dato una distribuzione normale (gaussiana), in modo da vedere cosa cosa mi veniva fuori
"""
#Shuffle the order of elements in a tensor
"""
not_shuffled = tf.constant([[10,7],
                           [3,4],
                           [2,5]])

shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)

tf.random.set_seed(42) #global seed
shuffled2 = tf.random.shuffle(not_shuffled, seed=42) #operation level random seed
print(shuffled2)
"""
#Creating tensors from NumPy arrays
"""
import numpy as np

x = tf.ones([10,7])
y = tf.zeros(shape=(4,3))
print(x, y)

numpy_A = np.arange(1,25, dtype=np.int32)
print(numpy_A)

A = tf.constant(numpy_A)
print(A)
B = tf.constant(numpy_A, shape=(2,3,4))
print(B)
C = tf.constant(numpy_A, shape=(8,3))
print(C)
"""
#Tensor attributes
"""
rank_4_tensor = tf.zeros(shape=[2,3,4,5])
print(rank_4_tensor) #2 matrici, con dentro 3 matrici ciascuna composte da 4 vettori di 5 elementi

print(rank_4_tensor[0]) #mi mostra solo la prima matrice delle due
print(rank_4_tensor[1][1][3][2]) #tipo questo mi mostrerà solo uno 0
print(rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor))
print(rank_4_tensor.shape[-1]) 
"""
#indexing and expanding tensors
"""
rank_4_tensor = tf.zeros(shape=[2,3,4,5])
x = rank_4_tensor[:2,:2,:2,:2]
y = rank_4_tensor[:1,:1,:1]
z = rank_4_tensor[:1,:1, :,:1]
print(x,y,z)

rank_2_tensor = tf.constant([[10,7],[3,4]])
za = rank_2_tensor[:,-1]
print(za)

rank_2_tensor_expanded = rank_2_tensor[..., tf.newaxis]
print(rank_2_tensor_expanded)
    #(alternativa)
other_rank_2_tensor_expanded = tf.expand_dims(rank_2_tensor, axis = -1)
print(other_rank_2_tensor_expanded)
"""
#manipulating Tensor with basic algebra
"""
tensor = tf.constant([[10,7],[3,4]])

print(tensor + 10)
print(tensor * 100)
print(tensor - 10)
print(tf.multiply(tensor,10)) #meglio usare i mnetodi di tensorflow per incrementare le performance
"""
#matrix manipulation with tensor 
"""
tensor = tf.constant([[10,7],[3,4]])

x= tf.matmul(tensor, tensor)
print(x) #dot notation con se stessa 

tensor_1 = tf.constant([[1,2,5],
                        [7,2,1],
                        [3,3,3]])
tensor_2 = tf.constant([[3,5],
                        [6,7],
                        [1,8]])
y = tf.matmul(tensor_1, tensor_2) #potevo anche riscvriverlo come tensor_1 @ tensor_2
print(y) #ovviamente per la dot notation valgono le stesse regole di numpy

tensor_3 = tf.reshape(tensor_2, shape=(2,3))
trasposta_tensor_3 = tf.transpose(tensor_3) 
print(tensor_3, trasposta_tensor_3) 

z= tf.matmul(tensor_2, tensor_3)
# za = tf.matmul(tensor_2, trasposta_tensor_3) ovviamente la dot notation in questo caso non funziona perchè usando la trasposta il numero di colonne e di righe va a cambiare (se avessi avuto una matrice quadrata avrebbe funzionato ma tra il reshape e la trasposta avrei semplicemente avuto i numeri in una disposizione diversa all'interno della matrice)
print(z) 

za= tf.tensordot(tensor_2, tensor_3, axes= 2)
print(za)
"""
#changing the datatype of tensors
"""
B = tf.constant([1.7,7.4])
print(B.dtype) # <dtype: 'float32'>

C = tf.constant([7,10])
print(C.dtype) # <dtype: 'int32'>

B_cast = tf.cast(B, dtype = tf.float16)
print(B_cast.dtype) #<dtype: 'float16'>

C_cast = tf.cast(C, dtype = tf.float64)
print(C_cast.dtype) #<dtype: 'float64'>
"""
#tensor aggregations

D = tf.constant([-7,-10])
D_abs = tf.abs(D)
print(D_abs)


tensor_random = tf.random.uniform(shape=(2,3,3), minval=0, maxval=10,seed = 42)
print(tensor_random)

tensor_random_min =tf.reduce_min(tensor_random)
tensor_random_max =tf.reduce_max(tensor_random)
tensor_random_mean =tf.reduce_mean(tensor_random)
tensor_random_sum =tf.reduce_sum(tensor_random)
tensor_random_variance =tf.math.reduce_variance(tensor_random)
tensor_random_stddev =tf.math.sqrt(tensor_random_variance)

print(tensor_random_min,tensor_random_max,tensor_random_mean,tensor_random_sum,tensor_random_variance, tensor_random_stddev)
