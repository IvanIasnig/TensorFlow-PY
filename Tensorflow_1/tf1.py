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

not_shuffled = tf.constant([[10,7],
                           [3,4],
                           [2,5]])

shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)

shuffled2 = tf.random.shuffle(not_shuffled, seed=42)
print(shuffled2)