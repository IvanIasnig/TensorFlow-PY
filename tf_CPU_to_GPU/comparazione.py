import tensorflow as tf
import numpy as np
import time

# Genera dati di esempio
num_classes = 10
x_train = np.random.random((100, 224, 224, 3))
y_train = np.random.randint(num_classes, size=(100, 1))
x_test = np.random.random((20, 224, 224, 3))
y_test = np.random.randint(num_classes, size=(20, 1))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Crea il modello ResNet-50
def create_model():
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Addestra il modello sulla GPU
print("Addestramento del modello ResNet-50 sulla GPU...")
with tf.device('/GPU:0'):
    model_gpu = create_model()
    start_time = time.time()
    model_gpu.fit(x_train, y_train,
                  epochs=5,
                  batch_size=16,
                  validation_data=(x_test, y_test),
                  verbose=0)
    gpu_time = time.time() - start_time
    print("Tempo impiegato per l'addestramento sulla GPU: {:.2f} secondi".format(gpu_time))

# Addestra il modello sulla CPU
print("Addestramento del modello ResNet-50 sulla CPU...")
with tf.device('/CPU:0'):
    model_cpu = create_model()
    start_time = time.time()
    model_cpu.fit(x_train, y_train,
                  epochs=5,
                  batch_size=16,
                  validation_data=(x_test, y_test),
                  verbose=0)
    cpu_time = time.time() - start_time
    print("Tempo impiegato per l'addestramento sulla CPU: {:.2f} secondi".format(cpu_time))



