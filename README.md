Diabetes Prediction Web Application
This project is a Flask-based web application that predicts the likelihood of diabetes based on user input. The app uses a support vector machine (SVM) model trained on a diabetes dataset to provide predictions. The model is capable of offering personalized diet plans based on the predicted probability of having diabetes.

Features
Prediction Page: Users can input their health data (e.g., glucose levels, skin thickness, insulin levels, and age) to get a prediction on the probability of having diabetes. Based on the prediction, the app suggests a customized diet plan.

Metrics Page: Displays the model's accuracy, confusion matrix, and classification report using the dataset, providing insights into the model’s performance.

Machine Learning Model: The backend uses an SVM model, trained on relevant features like glucose, skin thickness, insulin, and age, to predict diabetes outcomes.

Technologies Used
Flask: Web framework used to build the front-end and backend of the application.
scikit-learn: Used for data preprocessing, model training, and evaluation.
SVM (Support Vector Machine): Machine learning algorithm used for classification.
Pandas: Data manipulation and analysis.
NumPy: For numerical operations.
Matplotlib & Seaborn: Data visualization, especially for plotting the confusion matrix.
Pickle: For saving and loading the trained model and scaler.
