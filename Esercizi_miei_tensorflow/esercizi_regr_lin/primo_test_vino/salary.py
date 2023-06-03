import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('./redwine.csv', delimiter=';') 

X = wine.drop("quality", axis=1)
y = wine["quality"]

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

tf.random.set_seed(42)

wine_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

wine_model.compile(loss=tf.keras.losses.mse,
                          optimizer=tf.keras.optimizers.Adam(), 
                          metrics=['mse'])

history = wine_model.fit(X_train, y_train, epochs=200)

print(wine_model.evaluate(X_test, y_test))

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");
plt.show();

# Predicting on the test data
test_predictions = wine_model.predict(X_test).flatten()

for i in range(30):
    print("Real quality: ", y_test.iloc[i])
    print("Predicted quality: ", test_predictions[i])
    print("---")


new_wine = pd.DataFrame({
    "fixed acidity": [7.5],
    "volatile acidity": [0.2],
    "citric acid": [0.4],
    'residual sugar': [2.0],
    'chlorides': [0.04],
    'free sulfur dioxide': [45],
    'total sulfur dioxide': [140],
    'density': [0.992],
    'pH': [0.35],
    'sulphates': [0.6],
    'alcohol': [1.5]
})


prediction = wine_model.predict(new_wine)

print("wine quality: ", prediction[0][0]) 