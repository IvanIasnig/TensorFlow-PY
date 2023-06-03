import tensorflow as tf
from sklearn.datasets import make_circles
import datetime

print(tf.__version__)

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, 
                    noise=0.03, 
                    random_state=42)


# Check out the features
print(X)


# See the first 10 labels
y[:10]

# Make dataframe of features and labels
import pandas as pd
circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
circles.head()

# Check out the different labels
circles.label.value_counts()

# Visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
plt.show()