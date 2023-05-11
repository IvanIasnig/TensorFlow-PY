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