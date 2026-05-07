# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd


# =========================
# PROFESSIONAL UI CSS
# =========================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f5f7fb;
    font-family: 'Segoe UI', sans-serif;
}

/* Header Box */
.main-title {
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
}

.main-title h1 {
    font-family: 'Brush Script MT', cursive;
    font-size: 42px;
    margin-bottom: 5px;
}

.main-title p {
    font-size: 18px;
    opacity: 0.9;
}

/* Card Design */
.block-container {
    padding-top: 2rem;
}

/* Input Labels */
label {
    font-weight: 600 !important;
    color: #333 !important;
}

/* Inputs */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border: none;
    border-radius: 12px;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #ff3333, #ff5555);
}

/* Prediction Result Box */
.result-box {
    background: white;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
    text-align: center;
}

.result-box h2 {
    color: #4CAF50;
    margin-bottom: 10px;
}

.result-box h1 {
    color: #222;
    font-size: 50px;
}

/* Success Message */
.success-box {
    background: #d4edda;
    color: #155724;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    text-align: center;
    font-weight: bold;
    font-family: 'Brush Script MT', cursive;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# HELPER: EXTRACT OPTIONS FROM TRAINED COLUMNS
# =========================
def get_options(prefix):
    opts = [col.replace(prefix, "") for col in columns if col.startswith(prefix)]
    opts = sorted(list(set(opts)))
    return opts

# Extract all possible options
job_options = get_options("job_title_")
edu_options = get_options("education_level_")
loc_options = get_options("location_")
ind_options = get_options("industry_")
company_options = get_options("company_size_")
remote_options = get_options("remote_work_")

# Add baseline category (lost due to drop_first=True)
job_options = ["Other"] + job_options
edu_options = ["Other"] + edu_options
loc_options = ["Other"] + loc_options
ind_options = ["Other"] + ind_options
company_options = ["Other"] + company_options
remote_options = ["Other"] + remote_options

# =========================
# TITLE
# =========================
st.markdown("""
<div class="main-title">
    <h1>💼 Developer Salary Predictor</h1>

    <p style="
        font-size:18px;
        margin-top:10px;
        color:white;
    ">
        AI-Powered Salary Prediction Dashboard
    </p>

</div>
""", unsafe_allow_html=True)

# =========================
# USER INPUT
# =========================

col1, col2 = st.columns(2)

with col1:
    exp = st.number_input("Experience (years)", 0, 30)
    skills = st.number_input("Skills Count", 0, 50)
    cert = st.number_input("Certifications", 0, 20)

with col2:
    job = st.selectbox("Job Role", job_options)
    edu = st.selectbox("Education", edu_options)
    loc = st.selectbox("Location", loc_options)

ind = st.selectbox("Industry", ind_options)
company = st.selectbox("Company Size", company_options)
remote = st.selectbox("Remote Work", remote_options)

# =========================
# CREATE INPUT
# =========================
input_dict = {
    "experience_years": exp,
    "skills_count": skills,
    "certifications": cert,
    "job_title": job,
    "education_level": edu,
    "location": loc,
    "industry": ind,
    "company_size": company,
    "remote_work": remote
}

input_df = pd.DataFrame([input_dict])

# =========================
# FEATURE ENGINEERING
# =========================
input_df['exp_squared'] = input_df['experience_years'] ** 2
input_df['skill_per_exp'] = input_df['skills_count'] / (input_df['experience_years'] + 1)
input_df['cert_per_skill'] = input_df['certifications'] / (input_df['skills_count'] + 1)

input_df['seniority'] = pd.cut(
    input_df['experience_years'],
    bins=[0, 2, 5, 10, 20],
    labels=['Fresher', 'Junior', 'Mid', 'Senior']
)

# =========================
# DUMMIES + ALIGN
# =========================
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# =========================
# SCALE
# =========================
num_cols = ['experience_years', 'skills_count', 'certifications',
            'exp_squared', 'skill_per_exp', 'cert_per_skill']

input_df[num_cols] = scaler.transform(input_df[num_cols])

# =========================
# PREDICTION
# =========================
if st.button("Predict Salary"):

    prediction = model.predict(input_df)

    # Success Message
    st.markdown("""
    <div class="success-box">
        ✅ Prediction Completed Successfully
    </div>
    """, unsafe_allow_html=True)

    # Prediction Result Card
    st.markdown(f"""
    <div class="result-box">
        <h2>💰 Predicted Salary</h2>
        <h1 style="color:#4CAF50;">₹ {int(prediction[0]):,}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Animations
    st.balloons()
    st.snow()
