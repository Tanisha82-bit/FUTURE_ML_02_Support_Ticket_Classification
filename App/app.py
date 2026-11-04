import os
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Customer Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Theme Selector & Navigation
with st.sidebar:
  st.image("https://img.icons8.com/color/96/customer-support.png", width=65)
  st.markdown("### **Support AI Hub**")
  st.markdown("---")

  # Theme Mode Selector
  theme_mode = st.selectbox(
      "🎨 Appearance Theme",
      [
          "System Default",
          "Light Mode",
          "Dark Mode",
          "Ocean Blue",
          "Emerald Mint",
      ],
  )

  st.markdown("---")
  navigation = st.radio(
      "Navigation Menu",
      ["🎯 Live Classifier", "📊 Model Performance", "📁 Dataset Insights"],
      label_visibility="collapsed",
  )
  st.markdown("---")
  st.markdown(
      "<p style='font-size: 0.8rem; color: gray;'>Machine Learning"
      " Internship<br>Task 02 Project</p>",
      unsafe_allow_html=True,
  )

# Apply Dynamic CSS based on Selected Theme
theme_css = ""
if theme_mode == "Light Mode":
  theme_css = """
    <style>
    .stApp { background-color: #FFFFFF; color: #111827; }
    section[data-testid="stSidebar"] { background-color: #F3F4F6; }
    </style>
    """
elif theme_mode == "Dark Mode":
  theme_css = """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    </style>
    """
elif theme_mode == "Ocean Blue":
  theme_css = """
    <style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    section[data-testid="stSidebar"] { background-color: #1E293B; }
    </style>
    """
elif theme_mode == "Emerald Mint":
  theme_css = """
    <style>
    .stApp { background-color: #064E3B; color: #ECFDF5; }
    section[data-testid="stSidebar"] { background-color: #065F46; }
    </style>
    """

if theme_css:
  st.markdown(theme_css, unsafe_allow_html=True)


# Load and Train Models Cached
@st.cache_resource
def load_and_train_models():
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.pipeline import Pipeline
  from sklearn.svm import LinearSVC

  df = pd.read_csv("customer_support_tickets.csv")
  df["Ticket Subject"] = df["Ticket Subject"].fillna("").astype(str)
  df["Ticket Description"] = df["Ticket Description"].fillna("").astype(str)
  df["Text"] = (
      df["Ticket Subject"].str.strip() + " " + df["Ticket Description"].str.strip()
  )

  for col in ["Ticket Type", "Ticket Priority"]:
    df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
  df = df.dropna(subset=["Ticket Type", "Ticket Priority"]).copy()
  df = df[df["Text"].str.len() > 0].copy()

  X = df["Text"]
  y_type = df["Ticket Type"]
  y_priority = df["Ticket Priority"]

  cat_pipe = Pipeline([
      (
          "tfidf",
          TfidfVectorizer(
              lowercase=True,
              strip_accents="unicode",
              ngram_range=(1, 2),
              min_df=2,
              max_features=20000,
              sublinear_tf=True,
          ),
      ),
      (
          "classifier",
          LinearSVC(class_weight="balanced", random_state=42),
      ),
  ])
  cat_pipe.fit(X, y_type)

  pri_pipe = Pipeline([
      (
          "tfidf",
          TfidfVectorizer(
              lowercase=True,
              strip_accents="unicode",
              ngram_range=(1, 2),
              min_df=2,
              max_features=20000,
              sublinear_tf=True,
          ),
      ),
      (
          "classifier",
          LinearSVC(class_weight="balanced", random_state=42),
      ),
  ])
  pri_pipe.fit(X, y_priority)

  return cat_pipe, pri_pipe, df


cat_model, pri_model, df = load_and_train_models()

# Main Title Header
st.title("🎫 Customer Support Ticket Intelligence")
st.markdown(
    "Automated ticket categorization and priority assessment powered by"
    " Machine Learning (Linear SVM & TF-IDF)."
)
st.markdown("---")

if navigation == "🎯 Live Classifier":
  st.subheader("Test Customer Support Tickets in Real-Time")
  st.write(
      "Select a preset scenario or enter custom customer details below to"
      " instantly predict the ticket type and priority level."
  )

  col_sel1, col_sel2 = st.columns([1, 2])
  with col_sel1:
    template_choice = st.selectbox(
        "Scenario Templates",
        [
            "Custom Input",
            "Account Locked / Login Issue",
            "Refund Request for Delayed Shipping",
            "Billing Overcharge Complaint",
            "Product Warranty Inquiry",
        ],
    )

  templates = {
      "Custom Input": (
          "Enter subject...",
          "Enter detailed customer issue description here...",
      ),
      "Account Locked / Login Issue": (
          "Account Locked After Multiple Failed Logins",
          "My account has been locked after multiple failed login attempts. I"
          " cannot sign in and need help restoring access immediately.",
      ),
      "Refund Request for Delayed Shipping": (
          "Where is my refund for delayed order?",
          "I requested a refund last week for my order that never arrived, but"
          " I haven't seen any update on my account status.",
      ),
      "Billing Overcharge Complaint": (
          "Incorrect charge on monthly subscription invoice",
          "I was charged twice for my subscription this month. Please"
          " investigate and issue a credit to my payment method.",
      ),
      "Product Warranty Inquiry": (
          "Warranty coverage question for electronic device",
          "Does the extended warranty cover accidental water damage for the"
          " electronic device I purchased last month?",
      ),
  }

  default_sub, default_desc = templates[template_choice]

  sub_input = st.text_input("Ticket Subject Line", value=default_sub)
  desc_input = st.text_area("Ticket Description", value=default_desc, height=130)

  col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
  with col_btn1:
    classify_btn = st.button(
        "🚀 Run Classification", type="primary", use_container_width=True
    )

  if classify_btn:
    combined_text = f"{sub_input} {desc_input}".strip()
    if not combined_text:
      st.warning("Please provide valid ticket text.")
    else:
      pred_type = cat_model.predict([combined_text])[0]
      pred_priority = pri_model.predict([combined_text])[0]

      st.success("✨ Classification Successful!")

      res1, res2 = st.columns(2)
      with res1:
        st.metric(label="🏷️ Predicted Ticket Category", value=pred_type)
      with res2:
        st.metric(label="⚡ Predicted Priority Level", value=pred_priority)

elif navigation == "📊 Model Performance":
  st.subheader("Model Evaluation & Performance Benchmarks")
  st.write(
      "Review cross-validation scores, test metrics, and classification"
      " summaries generated during Task 02."
  )

  if os.path.exists("final_model_summary.csv"):
    summary_df = pd.read_csv("final_model_summary.csv")
    st.markdown("#### 🏆 Final Selected Models Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

  if os.path.exists("model_comparison_results.csv"):
    comparison_df = pd.read_csv("model_comparison_results.csv")
    st.markdown("#### 📈 Model Leaderboard & Comparison")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

  st.markdown("#### 📉 Confusion Matrices")
  c1, c2 = st.columns(2)
  with c1:
    if os.path.exists("images/category_confusion_matrix.png"):
      st.image(
          "images/category_confusion_matrix.png",
          caption="Ticket Type Confusion Matrix",
      )
  with c2:
    if os.path.exists("images/priority_confusion_matrix.png"):
      st.image(
          "images/priority_confusion_matrix.png",
          caption="Ticket Priority Confusion Matrix",
      )

elif navigation == "📁 Dataset Insights":
  st.subheader("Dataset Explorer")
  st.write(f"Loaded **{df.shape[0]:,}** clean customer support records.")
  st.dataframe(df.head(100), use_container_width=True)