import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("logreg_model.pkl")

st.set_page_config(page_title="Depression Prediction Survey", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton button { background-color: #6c63ff; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Social Media & Mental Health Survey")
st.markdown("Please fill in your details below to assess your mental wellness risk based on your social media habits.")

with st.form(key="survey_form"):
    age = st.number_input("Your Age", min_value=10, max_value=100)
    gender = st.selectbox("Gender", ['Male', 'Female', 'other'])
    relationship = st.selectbox("Relationship Status", ['Single', 'In a relationship', 'Married', 'Divorced'])
    occupation = st.selectbox("Occupation", ['University Student', 'Salaried Worker', 'Unemployed', 'School Student', 'Retired'])
    affiliate = st.selectbox("Affiliated Organization", ['University', 'Private', 'School', 'Company', 'Goverment', 'Other'])

    social_media_use = st.selectbox("Do you use social media?", ['Yes', 'No'])
    avg_time_per_day = st.selectbox("Average time spent daily on social media", [
        "Less than an Hour", "Between 1 and 2 hours", "Between 2 and 3 hours",
        "Between 3 and 4 hours", "Between 4 and 5 hours", "More than 5 hours"
    ])

    without_purpose = st.slider("Use without purpose", 0, 5, 2)
    distracted = st.slider("Easily distracted", 0, 5, 2)
    restless = st.slider("Feel restless when not online", 0, 5, 2)
    distracted_ease = st.slider("Easily distracted from tasks", 0, 5, 2)
    worries = st.slider("Experience worry/anxiety", 0, 5, 2)
    concentration = st.slider("Difficulty concentrating", 0, 5, 2)
    compare_to_others = st.slider("Compare yourself to others", 0, 5, 2)
    compare_feelings = st.slider("Compare feelings to others", 0, 5, 2)
    validation = st.slider("Seek validation online", 0, 5, 2)
    daily_activity_flux = st.slider("Fluctuation in daily activity", 0, 5, 2)
    sleeping_issues = st.slider("Trouble sleeping", 0, 5, 2)

    st.markdown("### Platforms You Use")
    platform_list = ['Instagram',
 'Snapchat',
 'LinkedIn',
 'Pinterest',
 'TikTok',
 'YouTube',
 'Twitter',
 'Reddit',
 'Facebook',
 'Discord']
    platform_inputs = {platform: st.checkbox(platform) for platform in platform_list}

    submit = st.form_submit_button("Predict Depression Risk")

if submit:
    gender_map = {'Male': 0, 'Female': 1, 'other': 2}
    relationship_map = {'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3}
    occupation_map = {'University Student': 0, 'Salaried Worker': 1, 'Unemployed': 2, 'School Student': 3, 'Retired': 4}
    affiliate_map = {'University': 0, 'Private': 1, 'School': 2, 'Company': 3, 'Goverment': 4, 'Other': 7}
    time_map = {
        "Less than an Hour": 0, "Between 1 and 2 hours": 1, "Between 2 and 3 hours": 2,
        "Between 3 and 4 hours": 3, "Between 4 and 5 hours": 4, "More than 5 hours": 5
    }

    input_data = pd.DataFrame([[
        age,
        gender_map[gender],
        relationship_map[relationship],
        occupation_map[occupation],
        affiliate_map[affiliate],
        1 if social_media_use == "Yes" else 0,
        time_map[avg_time_per_day],
        without_purpose,
        distracted,
        restless,
        distracted_ease,
        worries,
        concentration,
        compare_to_others,
        compare_feelings,
        validation,
        daily_activity_flux,
        sleeping_issues
    ]], columns=[
        "age", "gender", "relationship", "occupation", "affiliate_organization", "social_media_use",
        "avg_time_per_day", "without_purpose", "distracted", "restless", "distracted_ease", "worries",
        "concentration", "compare_to_others", "compare_feelings", "validation", "daily_activity_flux", "sleeping_issues"
    ])

    for platform in platform_list:
        input_data[platform] = 1 if platform_inputs[platform] else 0

    input_data["platform_sum"] = sum([input_data[platform][0] for platform in platform_list])
    input_data["impact_sum"] = (
        without_purpose + distracted + restless + distracted_ease + worries + concentration +
        compare_to_others + compare_feelings + validation + daily_activity_flux + sleeping_issues
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🗾️ Result")
    if prediction == 1:
        st.error(f"⚠️ You are likely to experience depressive symptoms. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ You are not likely to experience depressive symptoms. (Confidence: {1 - probability:.2%})")