import os
import joblib
import streamlit as st
from src.data_pipeline import prepare_data
from src.model import train_model, load_model

st.set_page_config(page_title="AI Student Support System", layout="wide")

MODEL_PATH = "models/student_support_model.pkl"

def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    
    X_train, X_test, y_train, y_test = prepare_data("data/student_data.csv")
    model = train_model(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

st.title("IBM AI Student Support System")
st.subheader("Academic Risk Detection and Student Support Prediction")

st.markdown("""
This project predicts whether a student may need additional academic support based on key educational indicators.
It was developed by Bilal Kurabi as a professional AI portfolio project.
""")

with st.form("student_form"):
    st.write("Enter the student details below:")
    
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    assignment_score = st.slider("Assignment Score (%)", 0, 100, 75)
    quiz_score = st.slider("Quiz Score (%)", 0, 100, 70)
    participation = st.slider("Participation (%)", 0, 100, 65)
    study_hours = st.slider("Study Hours per Week", 0, 40, 10)

    submitted = st.form_submit_button("Predict Support Need")

if submitted:
    model = get_model()
    features = [[attendance, assignment_score, quiz_score, participation, study_hours]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        result = "Needs Academic Support"
        probability_text = round(probability[1] * 100, 2)
    else:
        result = "On Track"
        probability_text = round(probability[0] * 100, 2)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {probability_text}%")

    if prediction == 1:
        st.warning("""
The model suggests this student may benefit from:
- academic mentoring
- additional tutoring support
- structured study planning
- early intervention
""")
    else:
        st.success("""
The model suggests the student is currently performing well and is likely on track.
""")