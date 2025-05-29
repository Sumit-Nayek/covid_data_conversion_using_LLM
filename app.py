# # --- app.py ---
# import streamlit as st
# import pandas as pd
# from format_utils import convert_row_to_text
# import json
# st.set_page_config(page_title="Medical Textifier", layout="wide")
# st.title("Medical Textifier 🏥🧠")

# # Sidebar navigation
# page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning"])

# # File uploader
# uploaded_file = st.file_uploader("Upload your medical CSV file", type=["csv"])
# if not uploaded_file:
#     st.warning("Please upload a CSV file to continue.")
#     st.stop()

# try:
#     df = pd.read_csv(uploaded_file)
#     df.fillna('', inplace=True)
# except Exception as e:
#     st.error(f"Error loading CSV: {e}")
#     st.stop()

# # Page 1: Transform Data
# if page == "🔄 Transform Data":
#     st.subheader("Step 1: Convert tabular data to structured clinical text")

#     required_columns = ['underlying_medical_condition']
#     missing_cols = [col for col in required_columns if col not in df.columns]

#     if missing_cols:
#         st.error(f"Missing required columns: {', '.join(missing_cols)}")
#     else:
#         df['text'] = df.apply(convert_row_to_text, axis=1)
#         st.success("✅ Text generation complete!")

#         st.dataframe(df[['text']])

#         # Optional: Add a 'label' column if available (e.g., final_test_result or generated_result)
#         if 'final_test_result' in df.columns:
#             df['label'] = df['final_test_result']

        
        
#         # Let the user choose output format
#         export_format = st.selectbox("Choose download format", ["CSV", "JSONL"])
        
#         if export_format == "CSV":
#             st.download_button(
#                 "📥 Download LLM-Ready CSV",
#                 data=df[['text', 'label']].to_csv(index=False).encode('utf-8') if 'label' in df.columns else df[['text']].to_csv(index=False).encode('utf-8'),
#                 file_name='llm_ready_dataset.csv',
#                 mime='text/csv'
#             )
        
#         elif export_format == "JSONL":
#             records = df[['text', 'label']].to_dict(orient='records') if 'label' in df.columns else df[['text']].to_dict(orient='records')
#             jsonl_data = "\n".join([json.dumps(record) for record in records])
#             st.download_button(
#                 "📥 Download LLM-Ready JSONL",
#                 data=jsonl_data.encode('utf-8'),
#                 file_name='llm_ready_dataset.jsonl',
#                 mime='application/json'
#             )
# # Page 2: Prepare for Fine-Tuning
# elif page == "📊 Prepare for Fine-Tuning":
#     st.subheader("Step 2: Fine-Tuning Analysis and Variable Prioritization")

#     st.markdown("This section helps identify key features for LLM-based fine-tuning based on rule-based scores (0–10).", unsafe_allow_html=True)

#     def simple_variable_score(col):
#         if df[col].nunique() == 2:
#             return 9
#         elif df[col].dtype == 'object':
#             return 7
#         elif df[col].dtype in ['int64', 'float64']:
#             return 5
#         return 3

#     scores = {col: simple_variable_score(col) for col in df.columns if col != 'Generated Text'}
#     score_df = pd.DataFrame(list(scores.items()), columns=["Feature", "LLM Score"]).sort_values(by="LLM Score", ascending=False)

#     st.dataframe(score_df)

#     top_features = score_df[score_df['LLM Score'] >= 7]['Feature'].tolist()
#     st.markdown("**Recommended features for fine-tuning:**")
#     st.markdown(", ".join(top_features) if top_features else "_No strong features identified._")

#     st.download_button(
#         "📅 Download Feature Priority CSV",
#         data=score_df.to_csv(index=False).encode('utf-8'),
#         file_name='llm_variable_scores.csv',
#         mime='text/csv'
#     )
import streamlit as st
import pandas as pd
from format_utils import convert_row_to_text
import json
st.set_page_config(page_title="Medical Textifier", layout="wide")
st.title("Medical Textifier 🏥🧠")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning"])

# File uploader
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

    required_columns = ['underlying_medical_condition']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
    else:
        df['text'] = df.apply(convert_row_to_text, axis=1)
        st.success("✅ Text generation complete!")

        st.dataframe(df[['text']])

        # Optional: Add a 'label' column if available (e.g., final_test_result or generated_result)
        if 'final_test_result' in df.columns:
            df['label'] = df['final_test_result']

        # Let the user choose output format
        export_format = st.selectbox("Choose download format", ["CSV", "JSONL"])
        
        if export_format == "CSV":
            st.download_button(
                "📥 Download LLM-Ready CSV",
                data=df[['text', 'label']].to_csv(index=False).encode('utf-8') if 'label' in df.columns else df[['text']].to_csv(index=False).encode('utf-8'),
                file_name='llm_ready_dataset.csv',
                mime='text/csv'
            )
        
        elif export_format == "JSONL":
            records = df[['text', 'label']].to_dict(orient='records') if 'label' in df.columns else df[['text']].to_dict(orient='records')
            jsonl_data = "\n".join([json.dumps(record) for record in records])
            st.download_button(
                "📥 Download LLM-Ready JSONL",
                data=jsonl_data.encode('utf-8'),
                file_name='llm_ready_dataset.jsonl',
                mime='application/json'
            )

# Page 2: Prepare for Fine-Tuning
elif page == "📊 Prepare for Fine-Tuning":
    st.subheader("Step 2: Fine-Tuning Analysis and Variable Prioritization")

    st.markdown("This section helps identify key features and recommend LLMs for fine-tuning based on your data and task objective.", unsafe_allow_html=True)

    # Feature 1: Objective Selection (Dropdown)
    task_objective = st.selectbox(
        "Select your task objective",
        ["Classification", "Text Generation"],
        help="Choose whether your goal is classification (e.g., predicting labels) or text generation (e.g., generating clinical notes)."
    )

    # Model Recommendation Logic
    def recommend_models(df, task_objective):
        dataset_size = len(df)
        has_labels = 'label' in df.columns or 'final_test_result' in df.columns
        avg_text_length = df['text'].apply(len).mean() if 'text' in df.columns else 0

        recommendations = []
        if task_objective == "Classification":
            if not has_labels:
                return ["Classification tasks require a 'label' or 'final_test_result' column in the dataset."]
            if dataset_size < 1000:
                recommendations.append("BERT (bert-base-uncased): Suitable for small datasets and binary/multiclass classification.")
                recommendations.append("DistilBERT: Lightweight, faster to train for smaller datasets.")
            elif dataset_size < 10000:
                recommendations.append("RoBERTa: Robust for medium-sized datasets with improved performance over BERT.")
                recommendations.append("ALBERT: Memory-efficient for classification tasks.")
            else:
                recommendations.append("DeBERTa: High performance for large datasets and complex classification tasks.")
                recommendations.append("Longformer: Suitable for long text sequences in classification.")
        else:  # Text Generation
            if avg_text_length > 500:
                recommendations.append("T5 (t5-base): Effective for text generation with long sequences.")
                recommendations.append("BART: Strong for tasks requiring text summarization or generation.")
            else:
                recommendations.append("GPT-2: Good for general text generation with smaller datasets.")
                recommendations.append("T5 (t5-small): Efficient for short text generation tasks.")
            if dataset_size > 10000:
                recommendations.append("LLaMA (via API): High performance for large-scale text generation (requires API access).")

        return recommendations if recommendations else ["No specific model recommendations based on the data and task."]

    # Existing Feature Scoring Logic
    def simple_variable_score(col):
        if df[col].nunique() == 2:
            return 9
        elif df[col].dtype == 'object':
            return 7
        elif df[col].dtype in ['int64', 'float64']:
            return 5
        return 3

    scores = {col: simple_variable_score(col) for col in df.columns if col != 'text'}
    score_df = pd.DataFrame(list(scores.items()), columns=["Feature", "LLM Score"]).sort_values(by="LLM Score", ascending=False)

    st.markdown("**Feature Prioritization for Fine-Tuning:**")
    st.dataframe(score_df)

    top_features = score_df[score_df['LLM Score'] >= 7]['Feature'].tolist()
    st.markdown("**Recommended features for fine-tuning:**")
    st.markdown(", ".join(top_features) if top_features else "_No strong features identified._")

    st.download_button(
        "📅 Download Feature Priority CSV",
        data=score_df.to_csv(index=False).encode('utf-8'),
        file_name='llm_variable_scores.csv',
        mime='text/csv'
    )

    # Feature 2: Button for Model Recommendations (Bottom Left)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add some spacing
    col1, col2 = st.columns([1, 3])  # Create two columns for layout control
    with col1:
        if st.button("Get LLM Recommendations"):
            st.markdown("**Recommended LLMs for Fine-Tuning:**")
            recommended_models = recommend_models(df, task_objective)
            for model in recommended_models:
                st.markdown(f"- {model}")
