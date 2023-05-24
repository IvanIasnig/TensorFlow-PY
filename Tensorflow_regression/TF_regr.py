#creating sample regression data
import tensorflow as tf
import matplotlib.pyplot as plt

"""
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0]) # Create features

y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]) # Create labels

print(plt.scatter(X, y)) # Visualize it
plt.show()

house_info = tf.constant(["bedroom", "bathroom", "garage"]) # Example input and output shapes of a regression model
house_price = tf.constant([939700])
print(house_info, house_price)
"""
#first regr AI
"""
# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])

# Fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=1000)

# Make a prediction with the model
output = model.predict([17.0])
print(output)

"""

#Steps in improving a model with TensorFlow

"""
# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Create the model 
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.Adam(lr=0.0001), #changed optimizer
              metrics=["mae"])

# Fit the model (for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=1000)

# Make a prediction with the model
output = model.predict([17.0])
print(output)
"""

#Steps in improving a model with TensorFlow 2/3

"""
# Create features (using tensors)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Create the model 
model = tf.keras.Sequential([
  tf.keras.layers.Dense(50, activation=None),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss="mae", #same thing rewritten
              optimizer=tf.keras.optimizers.Adam(lr=0.01), #changed optimizer
              metrics=["mae"])

# Fit the model (for longer)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

# Make a prediction with the model
output = model.predict([17.0])
print(output)
"""

# Evaluating a TensorFlow model part 1 -4
"""
import matplotlib.pyplot as plt

X = tf.range(-100, 100, 4)

y = X + 10
print(y)

#plt.plot(X, y)
#plt.show() #importante visualizzare il modello

print(len(X))

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

print(len(X_test),len(y_test),len(X_train),len(y_train))

plt.figure(figsize=(10,7))
plt.scatter(X_train, y_train, c="b", label="Training data")
plt.scatter(X_test, y_test, c="g", label="Testing data")
# plt.show()

#model.build() potevo usare questo metodo per farlo partire, ma è meglio dargli la input shape (senza non avrebbe funzionato)
#la shape indica che vogliamo un output per uninput rivecuto

tf.random.set_seed(42)

#creatre a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name="output_layer")
], name="one_of_many_model_we_will_build")

#compile a model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()
#output ->
#Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 1)                 2

# =================================================================
# Total params: 2
# Trainable params: 2
# Non-trainable params: 0
# _________________________________________________________________

model.fit(X_train, y_train, epochs=100, verbose=1)

from keras.utils import plot_model

plot_model(model, show_shapes=True)
"""

# Evaluating a TensorFlow model part 5/6/7

"""
import matplotlib.pyplot as plt
from keras.utils import plot_model #stesso codice di prima (ho tolto alcune cose di visualizzazione pura)

X = tf.range(-100, 100, 4)

y = X + 10
print(y)


X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]


#creatre a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer"),
    tf.keras.layers.Dense(1, name="output_layer")
], name="one_of_many_model_we_will_build")

#compile a model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()

model.fit(X_train, y_train, epochs=100, verbose=1)

plot_model(model, show_shapes=True)

y_pred = model.predict(X_test) #siamo andati ad agire solo sulla parte del test! Quindi i primi 40 numeri!
print(y_pred) 
print(y_test)


plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training data")
# Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Testing data")
# Plot the predictions in red (predictions were made on the test data)
plt.scatter(X_test, y_pred, c="r", label="Predictions")
# Show the legend
plt.legend();
plt.show();


mae = tf.keras.losses.mae(y_true=y_test, 
                                     y_pred=y_pred.squeeze())
print(mae)

mse = tf.keras.losses.mse(y_true=y_test, 
                                     y_pred=y_pred.squeeze())
print(mse)
"""

# Setting up TensorFlow modelling experiments part 1 (start with a simple model)
"""
X = tf.range(-100, 100, 4)

y = X + 10
print(y)


X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

y_preds_1 = model_1.predict(X_test)

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c="b", label="Training data")
plt.scatter(X_test, y_test, c="g", label="Testing data")
plt.scatter(X_test, y_preds_1, c="r", label="Predictions")
plt.legend();
plt.show();

def mae(y_test, y_pred):
  return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mse(y_test, y_pred):
  return tf.metrics.mean_squared_error(y_test,
                                       y_pred)

# Calculate model_1 metrics
mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
print(mae_1, mse_1)
"""