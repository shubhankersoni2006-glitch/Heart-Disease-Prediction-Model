import gradio as gr
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "🚨 The patient is **likely to have heart disease**."
    else:
        return "✅ The patient is **not likely to have heart disease**."

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex (1 = Male, 0 = Female)"),
        gr.Number(label="Chest Pain Type (0–3)"),
        gr.Number(label="Resting BP"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Fasting Blood Sugar > 120mg/dL (1=yes, 0=no)"),
        gr.Number(label="Resting ECG (0–2)"),
        gr.Number(label="Maximum Heart Rate Achieved"),
        gr.Number(label="Exercise Induced Angina (1=yes, 0=no)"),
        gr.Number(label="Oldpeak"),
        gr.Number(label="Slope of ST segment (0–2)"),
        gr.Number(label="Number of Major Vessels (0–3)"),
        gr.Number(label="Thalassemia (1–3)")
    ],
    outputs="text",
    title="Heart Disease Prediction App",
    description="Enter patient details to predict heart disease.",
)

iface.launch()
