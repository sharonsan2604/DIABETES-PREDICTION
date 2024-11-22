from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

# Load the model and the scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

def generate_prediction_sentence(input_data):
    prediction_prob = model.predict_proba(input_data)
    diabetes_probability = prediction_prob[0][1] * 100
    if diabetes_probability > 70:
        diet_plan = (
            f"The patient has a {diabetes_probability:.2f}% probability of having diabetes.\n"
            "Suggested diet plan:\n"
            "1. Include plenty of non-starchy vegetables like broccoli, spinach, and green beans.\n"
            "2. Choose whole grains like brown rice, quinoa, and whole wheat.\n"
            "3. Opt for lean proteins such as chicken, fish, and tofu.\n"
            "4. Incorporate healthy fats from avocados, nuts, and olive oil.\n"
            "5. Avoid sugary drinks and foods high in added sugars.\n"
            "6. Limit intake of processed foods and high-carb snacks."
        )
    elif diabetes_probability > 50:
        diet_plan = (
            f"The patient has a {diabetes_probability:.2f}% probability of having diabetes.\n"
            "Suggested diet plan:\n"
            "1. Focus on a balanced diet with plenty of vegetables and fruits.\n"
            "2. Choose whole grains over refined grains.\n"
            "3. Include a variety of protein sources, such as legumes, fish, and lean meat.\n"
            "4. Use healthy fats in moderation, such as olive oil and nuts.\n"
            "5. Avoid excessive intake of sugary beverages and high-sugar foods.\n"
            "6. Monitor portion sizes and avoid overeating."
        )
    else:
        diet_plan = (
            f"The patient has a {diabetes_probability:.2f}% probability of having diabetes.\n"
            "Suggested diet plan:\n"
            "1. Maintain a healthy, balanced diet with a variety of vegetables and fruits.\n"
            "2. Opt for whole grains like oats, barley, and brown rice.\n"
            "3. Include lean protein sources such as beans, lentils, and skinless poultry.\n"
            "4. Consume healthy fats from sources like fish, nuts, and seeds.\n"
            "5. Limit intake of sugary snacks and beverages.\n"
            "6. Stay active and incorporate regular physical exercise into daily routine."
        )

    return diet_plan

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        glucose = float(request.form['glucose'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        age = float(request.form['age'])
        
        input_data = sc.transform(np.array([[glucose, skin_thickness, insulin, age]]))
        prediction_sentence = generate_prediction_sentence(input_data)

        return render_template('result.html', result=prediction_sentence)
    except Exception as e:
        return str(e)

@app.route('/metrics')
def metrics():
    # Load the dataset
    dataset = pd.read_csv(r"C:\Users\Kanishaa\OneDrive\Documents\diabetes.csv")

    # Selecting relevant features and target variable
    X = dataset[['Glucose', 'SkinThickness', 'Insulin', 'Age']]
    Y = dataset['Outcome']

    # Normalizing the features
    X_scaled = sc.transform(X)

    # Evaluating the model
    accuracy = model.score(X_scaled, Y)

    # Generating predictions
    Y_pred = model.predict(X_scaled)

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(Y, Y_pred)
    class_report = classification_report(Y, Y_pred, output_dict=True)

    return render_template('metrics.html', accuracy=accuracy, conf_matrix=conf_matrix, class_report=class_report)

if __name__ == '__main__':
    app.run(debug=True)