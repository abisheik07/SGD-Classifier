# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1: Start
2: Load the Iris dataset and create a Pandas DataFrame with features and target.
3: Split the dataset into features (X) and target (y).
4: Split the data into training and testing sets using train_test_split.
5: Initialize the SGDClassifier with default parameters.
6: Train the classifier using the training data.

7: Predict the target values for the testing set.
8: Calculate and display the model's accuracy and confusion matrix.
9: End 
 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Abisheik Raj.J
RegisterNumber: 212224230006 
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
## Output:
![image](https://github.com/user-attachments/assets/77b4f5b2-c0ad-48c2-afc0-90c284ae8907)

![image](https://github.com/user-attachments/assets/47470ad2-19ee-470c-8871-a0432ee39e91)

![image](https://github.com/user-attachments/assets/765bf74b-241a-453f-bf0c-27c9b4908326)

![image](https://github.com/user-attachments/assets/07886f22-07aa-4806-9f23-d70159ff06f1)







## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
