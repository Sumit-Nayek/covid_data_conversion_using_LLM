import streamlit as st
import pandas as pd
from format_utils import convert_row_to_text

st.set_page_config(page_title="Medical Textifier", layout="wide")

st.title("Medical Textifier 🏥🧠")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning"])

# Shared CSV uploader
uploaded_file = st.file_uploader("Upload your medical CSV file", type=["csv"])

if not uploaded_file:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    df.fillna('', inplace=True)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Page 1: Transform Data
if page == "🔄 Transform Data":
    st.subheader("Step 1: Convert tabular data to structured clinical text")

    required_columns = ['comorbidity']  # Add more required columns as needed
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
    else:
        df['Generated Text'] = df.apply(convert_row_to_text, axis=1)
        st.success("✅ Text generation complete!")
        st.dataframe(df[['Generated Text']])

        st.download_button(
            "📥 Download Transformed CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='textified_output.csv',
            mime='text/csv'
        )

# Page 2: Prepare for Fine-Tuning
elif page == "📊 Prepare for Fine-Tuning":
    st.subheader("Step 2: Fine-Tuning Analysis and Variable Prioritization")

    st.markdown("📌 This section helps identify key features for LLM-based fine-tuning based on rule-based scores (0–10).")

    def simple_variable_score(col):
        if df[col].nunique() == 2:
            return 9  # Binary, easy for models
        elif df[col].dtype == 'object':
            return 7  # Textual, useful for NLP
        elif df[col].dtype in ['int64', 'float64']:
            return 5  # Numeric, can be embedded
        return 3  # Fallback

    scores = {col: simple_variable_score(col) for col in df.columns if col != 'Generated Text'}
    score_df = pd.DataFrame(list(scores.items()), columns=["Feature", "LLM Score"]).sort_values(by="LLM Score", ascending=False)

    st.dataframe(score_df)

    top_features = score_df[score_df['LLM Score'] >= 7]['Feature'].tolist()
    st.markdown("🔍 **Recommended features for fine-tuning:**")
    st.markdown(", ".join(top_features) if top_features else "_No strong features identified._")

    st.download_button(
        "📥 Download Feature Priority CSV",
        data=score_df.to_csv(index=False).encode('utf-8'),
        file_name='llm_variable_scores.csv',
        mime='text/csv'
    )
