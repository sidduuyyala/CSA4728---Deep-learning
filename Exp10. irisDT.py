# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model on the training data
classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12,8))
plot_tree(classifier, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, 
          rounded=True, 
          fontsize=6)

# Show the plot
plt.show()
