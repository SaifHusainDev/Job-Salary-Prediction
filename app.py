# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd
import json
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #192841;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #FFFFFF;
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #001f3f;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Input Box */
.stTextInput input,
.stNumberInput input {
    border-radius: 10px;
    border: 2px solid #2196F3;
}

/* Select Box */
.stSelectbox div {
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2196F3, #0D47A1);
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* Button Hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #1976D2, #0B3C91);
    transform: scale(1.02);
}

/* Success Box */
.stSuccess {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# USER DATABASE
# =========================
if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({}, f)

with open("users.json", "r") as f:
    users = json.load(f)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# HELPER FUNCTION
# =========================
def get_options(prefix):
    opts = [col.replace(prefix, "") for col in columns if col.startswith(prefix)]
    opts = sorted(list(set(opts)))
    return opts

# =========================
# OPTIONS
# =========================
job_options = ["Other"] + get_options("job_title_")
edu_options = ["Other"] + get_options("education_level_")
loc_options = ["Other"] + get_options("location_")
ind_options = ["Other"] + get_options("industry_")
company_options = ["Other"] + get_options("company_size_")
remote_options = ["Other"] + get_options("remote_work_")

# =========================
# TITLE
# =========================
st.title("💼 Salary Prediction App")

# =========================
# LOGIN / SIGNUP
# =========================
if not st.session_state.logged_in:

    if "signup_success" not in st.session_state:
        st.session_state.signup_success = False

    default_index = 0 if st.session_state.signup_success else 1

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Signup"],
        index=default_index
)

    # =========================
    # SIGNUP
    # =========================
    if menu == "Signup":

        st.subheader("Create Professional Account")

        first_name = st.text_input("First Name")
        sur_name = st.text_input("Surname")
        last_name = st.text_input("Last Name")
        email = st.text_input("Email Address")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")

        if st.button("Signup"):

            # Empty fields check
            if (
                first_name == "" or
                last_name == "" or
                email == "" or
                new_user == "" or
                new_pass == "" or
                confirm_pass == ""
            ):
                st.error("Please fill all fields")

            # Password match check
            elif new_pass != confirm_pass:
                st.error("Passwords do not match")

            # Existing user check
            elif new_user in users:
                st.error("Username already exists")

            else:

                users[new_user] = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "password": new_pass
                }

                with open("users.json", "w") as f:
                    json.dump(users, f)

                st.success("Account created successfully")
                st.info("Please login with your account")
                st.session_state.signup_success = True
                st.rerun()

    # =========================
    # LOGIN
    # =========================
    elif menu == "Login":

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username in users and users[username]["password"] == password:

                st.success("Login successful")
                st.session_state.logged_in = True
                st.rerun()

            else:
                st.error("Invalid username or password")

# =========================
# MAIN APP
# =========================
if st.session_state.logged_in:

    # Logout Button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.success("Welcome to Salary Prediction Dashboard")

    # =========================
    # USER INPUT
    # =========================
    exp = st.number_input("Experience (years)", 0, 30)
    skills = st.number_input("Skills Count", 0, 50)
    cert = st.number_input("Certifications", 0, 20)

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
    num_cols = [
        'experience_years',
        'skills_count',
        'certifications',
        'exp_squared',
        'skill_per_exp',
        'cert_per_skill'
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict Salary"):

        prediction = model.predict(input_df)

        st.success(f"💰 Predicted Salary: ₹ {int(prediction[0]):,}")

        st.balloons()
        st.snow()
