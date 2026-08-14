# ============================================================
# AI LOAN ELIGIBILITY PREDICTION + RAG DOCUMENT ASSISTANT
# Google Colab - Complete Single Cell Project
# ============================================================

!pip -q install gradio pypdf scikit-learn matplotlib

import pandas as pd
import numpy as np
import gradio as gr
import os
import re
import warnings
warnings.filterwarnings("ignore")

from pypdf import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE = "/content/train.csv"

if not os.path.exists(FILE):
    raise FileNotFoundError(
        "train.csv not found. Upload train.csv to Google Colab first."
    )

df = pd.read_csv(FILE)

print("Original Dataset Shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# 2. CLEAN DATASET
# ============================================================

# Keep only genuine target rows
df["Loan_Status"] = df["Loan_Status"].astype(str).str.strip()

df = df[df["Loan_Status"].isin(["Y", "N"])].copy()

# Convert numeric columns safely
numeric_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert categorical columns
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()

# Remove invalid rows
df = df.dropna(subset=["Loan_Status"])

print("\nClean Dataset Shape:", df.shape)
print("\nLoan Status:")
print(df["Loan_Status"].value_counts())


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

X = df[features]
y = df["Loan_Status"].map({"Y": 1, "N": 0})


# ============================================================
# 4. PREPROCESSING
# ============================================================

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, categorical_features),
    ("num", numerical_transformer, numerical_features)
])


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Eligible", "Eligible"],
    zero_division=0
))


# ============================================================
# 6. LOAN PREDICTION FUNCTION
# ============================================================

def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    income,
    co_income,
    loan_amount,
    loan_term,
    cibil,
    property_area
):

    # Convert CIBIL to credit history for the ML model
    if cibil >= 700:
        credit_history = 1.0
        credit_status = "Good"
    else:
        credit_history = 0.0
        credit_status = "Poor"

    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = model.predict(applicant)[0]

    probabilities = model.predict_proba(applicant)[0]

    # Probability for Eligible class
    eligible_probability = probabilities[1] * 100

    # --------------------------------------------------------
    # Project-level decision
    # --------------------------------------------------------

    if prediction == 1:

        result = "✅ ELIGIBLE"

        explanation = f"""
### Loan Eligibility Result

**Prediction:** ✅ ELIGIBLE

**Model confidence:** {eligible_probability:.2f}%

**Applicant Details**

| Feature | Value |
|---|---|
| Income | ₹{income:,.0f} |
| Co-applicant Income | ₹{co_income:,.0f} |
| Loan Amount | ₹{loan_amount:,.0f} |
| CIBIL Score | {cibil} |
| Credit History | {credit_status} |
| Education | {education} |
| Employment | {"Self Employed" if self_employed == "Yes" else "Salaried"} |
| Property Area | {property_area} |

### Why?

The trained ML model classified this application as **Eligible** based on the combination of the applicant's financial and demographic features.

The CIBIL score of **{cibil}** was converted to **Good credit history** for the model.

> This is an academic ML prediction and not an actual banking/loan approval.
"""

    else:

        result = "❌ NOT ELIGIBLE"

        explanation = f"""
### Loan Eligibility Result

**Prediction:** ❌ NOT ELIGIBLE

**Model confidence:** {(100 - eligible_probability):.2f}%

**Applicant Details**

| Feature | Value |
|---|---|
| Income | ₹{income:,.0f} |
| Co-applicant Income | ₹{co_income:,.0f} |
| Loan Amount | ₹{loan_amount:,.0f} |
| CIBIL Score | {cibil} |
| Credit History | {credit_status} |
| Education | {education} |
| Employment | {"Self Employed" if self_employed == "Yes" else "Salaried"} |
| Property Area | {property_area} |

### Result Explanation

The trained ML model classified this application as **Not Eligible** based on the combination of the available applicant features.

Possible contributing factors may include credit history, income, loan amount, loan term, and other applicant characteristics.

> This is an academic ML prediction and not an actual banking/loan rejection.
"""

    return explanation


# ============================================================
# 7. RAG DOCUMENT PROCESSING
# ============================================================

document_chunks = []


def process_pdf(pdf_file):

    global document_chunks

    if pdf_file is None:
        return "Please upload a PDF document."

    try:
        reader = PdfReader(pdf_file)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                full_text += text + "\n"

        if not full_text.strip():
            return "Could not extract text from this PDF."

        # Split document into chunks
        words = full_text.split()

        chunk_size = 180
        overlap = 40

        document_chunks = []

        start = 0

        while start < len(words):

            chunk = " ".join(
                words[start:start + chunk_size]
            )

            document_chunks.append(chunk)

            start += chunk_size - overlap

        return (
            f"✅ PDF processed successfully.\n\n"
            f"Pages: {len(reader.pages)}\n"
            f"Document chunks created: {len(document_chunks)}"
        )

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


# ============================================================
# 8. SIMPLE RAG RETRIEVAL
# ============================================================

def rag_question(question):

    if not document_chunks:
        return "Please upload and process a loan-policy PDF first."

    if not question.strip():
        return "Please enter a question."

    # Simple keyword-based retrieval
    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    scores = []

    for chunk in document_chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                chunk.lower()
            )
        )

        score = len(question_words.intersection(chunk_words))

        scores.append(score)

    # Get best chunks
    best_indices = np.argsort(scores)[-3:][::-1]

    best_chunks = [
        document_chunks[i]
        for i in best_indices
        if scores[i] > 0
    ]

    if not best_chunks:

        return "❌ Content not found in the uploaded document."

    answer = "\n\n".join(best_chunks)

    return f"""
### 📄 Answer from Uploaded Document

{answer}

---

**Source:** Uploaded loan document
"""


# ============================================================
# 9. DASHBOARD CSS
# ============================================================

css = """
.gradio-container {
    max-width: 1200px !important;
}

h1 {
    text-align: center;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
}
"""


# ============================================================
# 10. BUILD GRADIO DASHBOARD
# ============================================================

with gr.Blocks(
    title="AI Loan Eligibility System",
    css=css
) as app:

    gr.Markdown("""
# 🏦 AI Loan Eligibility Prediction System

### Machine Learning + RAG Loan Document Assistant

This system predicts loan eligibility using historical loan data
and answers questions from uploaded loan-policy documents.
""")

    # --------------------------------------------------------
    # TAB 1 - LOAN PREDICTION
    # --------------------------------------------------------

    with gr.Tab("🏦 Loan Eligibility Prediction"):

        gr.Markdown("""
## Enter Applicant Details

The ML model will evaluate the applicant using the features
available in the training dataset.
""")

        with gr.Row():

            with gr.Column():

                gender = gr.Dropdown(
                    ["Male", "Female"],
                    label="Gender",
                    value="Male"
                )

                married = gr.Dropdown(
                    ["Yes", "No"],
                    label="Married",
                    value="Yes"
                )

                dependents = gr.Dropdown(
                    ["0", "1", "2", "3+"],
                    label="Dependents",
                    value="0"
                )

                education = gr.Dropdown(
                    ["Graduate", "Not Graduate"],
                    label="Education",
                    value="Graduate"
                )

                self_employed = gr.Dropdown(
                    ["Yes", "No"],
                    label="Employment",
                    value="No"
                )

            with gr.Column():

                income = gr.Number(
                    label="Applicant Income (₹)",
                    value=50000
                )

                co_income = gr.Number(
                    label="Co-applicant Income (₹)",
                    value=0
                )

                loan_amount = gr.Number(
                    label="Loan Amount (₹)",
                    value=300000
                )

                loan_term = gr.Number(
                    label="Loan Term (months)",
                    value=360
                )

                cibil = gr.Number(
                    label="CIBIL Score",
                    value=750,
                    minimum=300,
                    maximum=900
                )

                property_area = gr.Dropdown(
                    ["Urban", "Semiurban", "Rural"],
                    label="Property Area",
                    value="Urban"
                )

        predict_button = gr.Button(
            "🔍 CHECK LOAN ELIGIBILITY",
            variant="primary"
        )

        prediction_output = gr.Markdown()

        predict_button.click(
            fn=predict_loan,
            inputs=[
                gender,
                married,
                dependents,
                education,
                self_employed,
                income,
                co_income,
                loan_amount,
                loan_term,
                cibil,
                property_area
            ],
            outputs=prediction_output
        )

        gr.Markdown(f"""
### 📊 Model Information

**Algorithm:** Random Forest Classifier

**Training records:** {len(df)}

**Features:** {len(features)}

**Test Accuracy:** {accuracy * 100:.2f}%

**Target:** Loan Approval (`Y` / `N`)

---

⚠️ **Academic Project Notice:**  
The prediction is based on the supplied dataset and is intended
for educational demonstration. It is not a real financial
approval/rejection decision.
""")


    # --------------------------------------------------------
    # TAB 2 - RAG DOCUMENT ASSISTANT
    # --------------------------------------------------------

    with gr.Tab("📄 Loan Document Assistant"):

        gr.Markdown("""
# 📄 RAG Loan Document Assistant

Upload a loan policy / eligibility PDF.

The system retrieves relevant content from the uploaded document
when you ask a question.
""")

        pdf_upload = gr.File(
            label="Upload Loan Policy PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        process_button = gr.Button(
            "📚 Process PDF",
            variant="primary"
        )

        process_status = gr.Markdown()

        process_button.click(
            fn=process_pdf,
            inputs=pdf_upload,
            outputs=process_status
        )

        question = gr.Textbox(
            label="Ask a question",
            placeholder="Example: What documents are required for a loan?"
        )

        ask_button = gr.Button(
            "🔎 Ask from Document"
        )

        rag_output = gr.Markdown()

        ask_button.click(
            fn=rag_question,
            inputs=question,
            outputs=rag_output
        )


    # --------------------------------------------------------
    # TAB 3 - DATASET DETAILS
    # --------------------------------------------------------

    with gr.Tab("📊 Dataset Dashboard"):

        gr.Markdown("# Dataset Overview")

        gr.Markdown(
            f"""
**Total valid records:** {len(df)}

**Eligible records:** {(df["Loan_Status"] == "Y").sum()}

**Not Eligible records:** {(df["Loan_Status"] == "N").sum()}

**Model Accuracy:** {accuracy * 100:.2f}%
"""
        )

        dataset_display = df.head(50)

        gr.Dataframe(
            value=dataset_display,
            label="Loan Dataset"
        )


# ============================================================
# 11. LAUNCH
# ============================================================

print("\n======================================")
print("AI LOAN ELIGIBILITY SYSTEM READY")
print("======================================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("Valid Dataset Records:", len(df))

app.launch(share=True)
