#creating sample regression data
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

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
# Setting up TensorFlow modelling experiments part 2 (increasing complexity)
"""
X = tf.range(-100, 100, 4)

y = X + 10
print(y)

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation= "relu"), #aggiunto hidden layer prima dell'output layer
  tf.keras.layers.Dense(1) #output layer come prima
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500) #aumentato il numero di epoches

y_preds_1 = model_2.predict(X_test)

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

# Calculate model_2 metrics

mae_2 = mae(y_test, y_preds_1.squeeze()).numpy()
mse_2 = mse(y_test, y_preds_1.squeeze()).numpy()
print(mae_1, mse_1)
"""
# Comparing and tracking your TensorFlow modelling experiments
"""
import pandas as pd


model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results)
"""
# How to save a TensorFlow model
"""
model_2.save('prova_salvataggio_modello')

model_2.save('prova_salvataggio_modello.h5') #modello salvato in hdf5 format

#How to load and use a saved TensorFlow model

loaded_saved_model = tf.keras.models.load_model("prova_salvataggio_modello")
print(loaded_saved_model.summary())

loaded_saved_model = tf.keras.models.load_model("prova_salvataggio_modello.h5")
print(loaded_saved_model.summary()) #ovviamente i summary dei due modelli sono uguali

model_2_preds = model_2.predict(X_test)
saved_model_preds = loaded_saved_model.predict(X_test)
print(mae(y_test, saved_model_preds.squeeze()).numpy() == mae(y_test, model_2_preds.squeeze()).numpy()) #questo pezzo di codice serve solo a vedere se l'import è corretto, ovviamente il modello e il modello salvato sono identici quindi abbiamo come risultato "true"
"""
# Putting together what we've learned part 1 (preparing a dataset)

pd.set_option('display.max_columns', None) #per fare in modo che pandas mi mostri tutte le colonne e non mi mostri i puntini di sospensione 

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
  #print(insurance) #[1338 rows x 7 columns]

insurance_one_hot = pd.get_dummies(insurance) #one hot encoding
print(insurance_one_hot.head()) # view the converted columns


# Create X & y values
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]


# Create training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) # set random state for reproducible splits
     
# View features
print(X.head())

# Set random seed
tf.random.set_seed(42)

# Add an extra layer and increase number of units
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 units
  tf.keras.layers.Dense(10), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

# Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't 
                          metrics=['mae'])

# Fit the model and save the history (we can plot this)
history = insurance_model_2.fit(X_train, y_train, epochs=115) #non serve riformattare X_train perchè pandas trasforma tutto in un array di numpy, quindi tensorflow sa già come gestirlo

# Evaluate our larger model
print(insurance_model_2.evaluate(X_test, y_test))

#prima di creare il primo modello su github era stato costruito un primo modello che però aveva 7000 di mae come risultato
#contando che la mediana è 9500 e la media 13300 (y_train.median() e y_train.mean()) il risultato era troppo sbagliato

# Plot history (also known as a loss curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");
plt.show() #guardando la curva della loss mi sono reso conto che il modello scende velocemente fino ad un addestramento di 115 epoches, poi scende piano 

# creiamo un nuovo dataframe con i dettagli del nuovo individuo
person_x = pd.DataFrame({'age': [30],
                         'bmi': [30],
                         'children': [1],
                         'sex_female': [1],
                         'sex_male': [0],
                         'smoker_no': [1],
                         'smoker_yes': [0],
                         'region_northeast': [1],
                         'region_northwest': [0],
                         'region_southeast': [0],
                         'region_southwest': [0]})

# usiamo il modello per fare la previsione
prediction = insurance_model_2.predict(person_x)

print("insurance cost: ", prediction[0][0])
