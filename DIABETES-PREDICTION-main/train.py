import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Kanishaa\OneDrive\Documents\diabetes.csv")

# Selecting relevant features and target variable
X = dataset[['Glucose', 'SkinThickness', 'Insulin', 'Age']]
Y = dataset['Outcome']

# Normalizing the features
sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])

# Training the model with probability estimates enabled
svc = SVC(kernel='linear', probability=True, random_state=42)
svc.fit(X_train, Y_train)

# Evaluating the model
accuracy = svc.score(X_test, Y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Generating predictions
Y_pred = svc.predict(X_test)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Saving the model
with open('model.pkl', 'wb') as file:
    pickle.dump(svc, file)
