import streamlit as st
import pandas as pd
import joblib
# Load the trained pipeline
model = joblib.load('student_performance_pipeline.pkl')

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Predictor")
st.write(
    "This tool predicts whether a student is likely to be a **top performer** "
    "(80%+ average across math, reading, and writing) based on demographic and "
    "preparatory factors. Use it as an early screening signal, not a final verdict."
)

st.divider()

# Input widgets — one per feature the model was trained on

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"]
    )
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

with col2:
    parent_edu = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
    )
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

st.divider()

# Prediction

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [parent_edu],
        'lunch': [lunch],
        'test preparation course': [test_prep],
    })

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    pass_confidence = probabilities[classes.index('pass')]
    fail_confidence = probabilities[classes.index('fail')]

    st.subheader("Result")

    if prediction == 'pass':
        st.success(f"Predicted: **Top Performer** ")
        st.metric("Confidence (Top Performer)", f"{pass_confidence:.1%}")
    else:
        st.warning(f"Predicted: **At Risk** ")
        st.metric("Confidence (At Risk)", f"{fail_confidence:.1%}")

    st.caption(
        "Note: this model uses demographic and preparatory factors only, and has "
        "moderate precision. Treat this as a screening signal to prompt further "
        "assessment — not a definitive outcome."
    )

    with st.expander("See full prediction probabilities"):
        prob_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
