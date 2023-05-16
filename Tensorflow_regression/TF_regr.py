#creating sample regression data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0]) # Create features

y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]) # Create labels

print(plt.scatter(X, y)) # Visualize it
plt.show()

house_info = tf.constant(["bedroom", "bathroom", "garage"]) # Example input and output shapes of a regression model
house_price = tf.constant([939700])
print(house_info, house_price)
     