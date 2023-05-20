#creating sample regression data
import tensorflow as tf

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

