# Step 1: Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (Petal length, Petal width, etc.)
y = iris.target  # Labels (0, 1, 2 representing different species)

# Step 3: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create an SVM classifier and train it
model = SVC(kernel='linear')  # You can change the kernel to 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = model.predict(X_test)

# Step 6: Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 7: Print the accuracy
print(f"Accuracy of the SVM model on the Iris dataset: {accuracy * 100:.2f}%")
