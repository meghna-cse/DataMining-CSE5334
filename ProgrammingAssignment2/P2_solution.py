"""
Author: Meghna J
Course: CSE 5334 Data Mining
Description: Solution for P2 assignment

Pre-requisite:
This script requires that both the datasets used, 'nba_stats.csv' and
'dummy_test.csv', are present in the same directory as this script.
"""


#Imported all the required libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('nba_stats.csv')

# Print unique position counts
print(data['Pos'].value_counts())
print(data.columns)

# Taking care of missing values using mean values
imputer = SimpleImputer(strategy='mean')                                # Create an imputer object with a mean filling strategy
numeric_columns = data.select_dtypes(include=['number']).columns        # Select numeric columns for imputation
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])    # Apply imputation

# Encoding
labelencoder = LabelEncoder()
for i in data.select_dtypes(np.object0):
    data[i] = labelencoder.fit_transform(data[i])

# Correlation matrix
corr_matrix = data.corr()  
target_correlations = corr_matrix['Pos'].abs().sort_values(ascending=False)
print(target_correlations)      # Print sorted correlation values to see how each feature correlates with 'Pos'

# Selecting Features to include based on correlation threshold of approximately 0.05
selected_features = ['ORB', 'BLK', '3PA', 'TRB', '3P', 'FG%', '3P%', 'DRB', '2P%', 'STL', 'PF', 'eFG%', '2P', 'AST', 'FGA', 'FT%', 'MP', '2PA', 'PTS', 'Age']
X = data[selected_features]
y = data['Pos']


# -------------------------------------------------------------------------------
# TASK 1:
## Dataset: nba_stats.csv
## Data Split: 80% training set, 20% validation set
## Classifier used: Neural Network using MLPClassifier
## Displaying accuracy and correlation matrix for training and validation sets
# -------------------------------------------------------------------------------

print()
print('---------------------Task 1-------------------------')

# Split the data into training and testing sets with 80% training and 20% testing
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Classifier
classifier = MLPClassifier(random_state=0, learning_rate_init=(0.001), hidden_layer_sizes=(150,), max_iter=400, activation='tanh', solver='adam')
classifier.fit(X_train, y_train)

# Predict on the training and validation sets
train_predictions = classifier.predict(X_train)
validation_predictions = classifier.predict(X_validation)

# Training and validation accuracies
train_accuracy = accuracy_score(y_train, train_predictions)
validation_accuracy = accuracy_score(y_validation, validation_predictions)
print('Training Accuracy:', train_accuracy*100, "%")
print('Validation Accuracy:', validation_accuracy*100, "%")

# Training and validation confusion matrices
print('Training Confusion Matrix:\n', confusion_matrix(y_train, train_predictions))
print('Validation Confusion Matrix:\n', confusion_matrix(y_validation, validation_predictions))


# -------------------------------------------------------------------------------
# TASK 2:
## Dataset: dummy_test.csv
## Classifier used: Neural Network using MLPClassifier
## Displaying accuracy and correlation matrix for the test set
# -------------------------------------------------------------------------------
print()
print('---------------------Task 2-------------------------')

# Load the dummy test set
test_data = pd.read_csv('dummy_test.csv')

# Using the same feature set as in task 1
dummy_test_features = test_data[selected_features]

# Encoding
labelencoder = LabelEncoder()
for i in test_data.select_dtypes(np.object0):
    test_data[i] = labelencoder.fit_transform(test_data[i])

dummy_test_labels = test_data['Pos']

dummy_predictions = classifier.predict(dummy_test_features)

# Calculate accuracy
dummy_accuracy = accuracy_score(dummy_test_labels, dummy_predictions)
print('Accuracy:', dummy_accuracy*100, "%")

# Generate and display confusion matrix
dummy_confusion_matrix = confusion_matrix(dummy_test_labels, dummy_predictions)
print('Confusion Matrix:\n', dummy_confusion_matrix)


# -------------------------------------------------------------------------------
# TASK 3:
## Dataset: nba_stats.csv
## Classifier used: Neural Network using MLPClassifier
## Method: 10-fold stratified cross-validation
## Displaying accuracy for each fold and the average accuracy
# -------------------------------------------------------------------------------
print()
print('---------------------Task 3-------------------------')

# 10-fold stratified cross-validation
cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cross_val_scores = cross_val_score(classifier, X, y, cv=cross_validation, scoring='accuracy')

# Print the accuracies

print("Accuracies for each fold:")
for i, acc in enumerate(cross_val_scores, 1):
    print(f"Fold {i}: {acc:.4f}")

print("In 10 Folds, Maximum Accuracy is ", cross_val_scores.max()*100, "% and Minimum Accuracy is ", cross_val_scores.min()*100, "%")
print("Average Accuracy:", cross_val_scores.mean()*100, "%")