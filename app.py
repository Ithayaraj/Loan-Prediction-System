import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for user-friendly UI
st.markdown("""
<style>
    /* Global Font Size Increase */
    html, body, [class*="css"] {
        font-family: 'sans-serif';
    }
    
    /* Headers */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #ff4b4b !important;
    }
    h3 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
    }
    p {
        font-size: 1.2rem !important;
    }

    /* Input Labels (Gender, Income, etc.) */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem !important; 
        font-weight: 500;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 1.3rem !important;
        color: #31333F !important;
    }

    /* Input Fields (The box itself) */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput input {
        min-height: 50px;
        font-size: 1.2rem !important;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 1.5rem !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border-color: #ff6b6b;
    }
    
    /* Sidebar and Container adjustments */
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 3rem;
        max-width: 1200px;
    }
    
    /* Toolbar / Header adjustments */
    header[data-testid="stHeader"] {
        height: 6rem !important; 
    }
    div[data-testid="stToolbar"] {
        top: 2rem !important; /* Move down slightly */
        right: 2rem !important; /* Move left slightly */
    }
    div[data-testid="stToolbar"] button {
        transform: scale(1.3) !important; 
        margin-left: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 Loan Approval Prediction System")
st.markdown("### 🏦 Check your eligibility in seconds")
st.markdown("Fill in the details below to get an instant loan approval decision.")
st.divider()

# ---------------- INPUTS ----------------

st.subheader("👤 Personal Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    married = st.selectbox("Married", ["Yes", "No"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col3:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()
st.subheader("💸 Financial Details")

f_col1, f_col2 = st.columns(2)

with f_col1:
    applicant_income = st.number_input(
        "Applicant Annual Income (LKR)", 
        min_value=0, 
        value=None, 
        step=10000,
        placeholder="Enter your annual income...",
        help="Total income you earn per year before tax"
    )
    if applicant_income:
        st.caption(f"**👁️ {applicant_income:,.0f} LKR**")

with f_col2:
    coapplicant_income = st.number_input(
        "Co-applicant Annual Income (LKR)", 
        min_value=0, 
        value=None, 
        step=10000,
        placeholder="Enter co-applicant income (0 if none)...",
        help="Income of spouse or partner if applying jointly"
    )
    if coapplicant_income:
        st.caption(f"**👁️ {coapplicant_income:,.0f} LKR**")

st.divider()
st.subheader("📋 Loan Request")

l_col1, l_col2, l_col3 = st.columns(3)

with l_col1:
    loan_amount = st.number_input(
        "Loan Amount (LKR)", 
        min_value=0, 
        value=None, 
        step=50000,
        placeholder="Enter desired amount...",
        help="How much money do you need?"
    )
    if loan_amount:
        st.caption(f"**👁️ {loan_amount:,.0f} LKR**")

with l_col2:
    loan_term = st.number_input(
        "Loan Term (Days)", 
        min_value=0, 
        value=None, 
        step=30,
        placeholder="Ex: 360 (1 year)",
        help="Duration of the loan in days"
    )

with l_col3:
    credit_type = st.selectbox(
        "Credit History",
        ["Good", "Bad", "Fresher (No History)"],
        help="Your past repayment behavior"
    )

# ---------------- ENCODING ----------------
# Safe handling of None values (if user hasn't typed yet)
app_income_val = applicant_income if applicant_income is not None else 0
coapp_income_val = coapplicant_income if coapplicant_income is not None else 0
loan_amt_val = loan_amount if loan_amount is not None else 0
loan_term_val = loan_term if loan_term is not None else 0

# Mapping inputs to model format
gender_enc = 1 if gender == "Male" else 0
married_enc = 1 if married == "Yes" else 0
education_enc = 1 if education == "Graduate" else 0
self_employed_enc = 1 if self_employed == "Yes" else 0

dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

dependents_enc = dependents_map[dependents]
property_area_enc = property_map[property_area]

# Credit History Logic
if credit_type == "Good":
    credit_history_enc = 1
elif credit_type == "Bad":
    credit_history_enc = 0
else:
    credit_history_enc = -1  # Flag for Fresher

# ---------------- PREDICTION ----------------
st.write("") # Spacer
if st.button("🔍 Predict Loan Status", use_container_width=True):

    # 1. Input Validation
    if app_income_val == 0 and coapp_income_val == 0:
        st.warning("⚠️ Please enter at least one source of income.")
        st.stop()
        
    if loan_amt_val == 0 or loan_term_val == 0:
        st.warning("⚠️ Please enter a valid Loan Amount and Term.")
        st.stop()

    total_annual_income = app_income_val + coapp_income_val

    # 2. Credit History Checks
    if credit_history_enc == 0:
        st.error("❌ Loan Rejected: Poor credit history")
        st.stop()

    # 3. Affordability Calculations
    daily_income = total_annual_income / 365
    daily_emi = loan_amt_val / loan_term_val

    # 4. Fresher vs Standard Logic
    if credit_history_enc == -1:  # Fresher
        # Rule: Freshers must have their OWN income
        if app_income_val <= 0:
            st.error("❌ Loan Rejected: Fresher without personal income is not eligible")
            st.stop()
        
        # Rule: Stricter EMI ratio for freshers (40%)
        if daily_emi > daily_income * 0.4:
            st.error(f"❌ Loan Rejected: Monthly EMI ({daily_emi*30:,.0f} LKR) is too high for your income level (Fresher Limit).")
            st.stop()
            
        # Boost to Good History if they pass checks
        credit_history_enc = 1 
    else:
        # Standard Applicants (50% ratio)
        if daily_emi > daily_income * 0.5:
            st.error(f"❌ Loan Rejected: Monthly EMI ({daily_emi*30:,.0f} LKR) exceeds affordability limits.")
            st.stop()

    # 5. Model Prediction
    input_data = pd.DataFrame([[  
        gender_enc,
        married_enc,
        dependents_enc,
        education_enc,
        self_employed_enc,
        app_income_val,
        coapp_income_val,
        loan_amt_val,
        loan_term_val,
        credit_history_enc,
        property_area_enc
    ]], columns=feature_names)

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.divider()
        if prediction[0] == 1:
            st.balloons()
            st.success("## ✅ Loan Approved!")
            st.markdown(f"**Congratulations!** You are eligible for the loan of **LKR {loan_amt_val:,.0f}**.")
            st.info(f"💰 Estimated Monthly Installment: **LKR {daily_emi*30:,.2f}**")
        else:
            st.error("## ❌ Loan Not Approved")
            st.markdown("Based on our analysis, we cannot approve this loan request at this time.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
