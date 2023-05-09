import tensorflow as tf
import time

# Crea un tensore di dimensione 1000x1000 e calcola il quadrato di ogni elemento
a = tf.random.normal([10000, 10000])
start_time = time.time()
b = tf.square(a)
c = tf.square(a)
d = tf.square(a)
e = tf.square(a)
f = tf.square(a)
g = tf.square(a)
end_time = time.time()
print(f"Tempo di esecuzione: {end_time - start_time} secondi")
