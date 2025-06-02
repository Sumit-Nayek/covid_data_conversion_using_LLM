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
# import matplotlib.pyplot as plt
from format_utils import convert_row_to_text
import json
import io
import uuid

st.set_page_config(page_title="Medical Textifier", layout="wide")
st.title("Medical Textifier 🏥🧠")

# Initialize session state for storing DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning"])

# File uploader
uploaded_file = st.file_uploader("Upload your medical CSV file", type=["csv"])
if uploaded_file:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.df.fillna('', inplace=True)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

# Page 1: Transform Data
if page == "🔄 Transform Data":
    st.subheader("Step 1: Convert tabular data to structured clinical text")

    if st.session_state.df is None:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

    df = st.session_state.df

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

        # Update session state with transformed DataFrame
        st.session_state.df = df

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

    if st.session_state.df is None:
        st.warning("Please upload a CSV file in the Transform Data page to continue.")
        st.stop()

    df = st.session_state.df

    st.markdown("This section helps identify key features and recommend LLMs for fine-tuning based on your data and task objective.", unsafe_allow_html=True)

    # Objective Selection (Dropdown)
    task_objective = st.selectbox(
        "Select your task objective",
        ["Classification", "Text Generation"],
        help="Choose whether your goal is classification (e.g., predicting labels) or text generation (e.g., generating clinical notes)."
    )

    # Feature Scoring Logic
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

    # Button for Visualizing Feature Scores
    st.markdown("**Feature Prioritization for Fine-Tuning:**")
    if st.button("Visualize Feature Scores"):
            st.write("Under construction....")
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.bar(score_df["Feature"], score_df["LLM Score"], color='skyblue')
        # ax.set_xlabel("Features")
        # ax.set_ylabel("LLM Score (0-10)")
        # ax.set_title("Feature Prioritization for LLM Fine-Tuning")
        # plt.xticks(rotation=45, ha='right')
        # plt.tight_layout()

        # # Save plot to a bytes buffer
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # st.image(buf, caption="Feature Score Visualization", use_column_width=True)
        # plt.close(fig)

    # top_features = score_df[score_df['LLM Score'] >= 2]['Feature'].tolist()
    # st.markdown("**Recommended features for fine-tuning:**")
    # st.markdown(", ".join(top_features) if top_features else "_No strong features identified._")

    # st.download_button(
    #     "📅 Download Feature Priority CSV",
    #     data=score_df.to_csv(index=False).encode('utf-8'),
    #     file_name='llm_variable_scores.csv',
    #     mime='text/csv'
    # )
import numpy as np
    # Model Recommendation Logic
def recommend_models(df, task_objective):
            dataset_size = len(df)
            text_column = 'text' if 'text' in df.columns else None
            avg_text_length = df[text_column].apply(lambda x: len(str(x).split())).mean() if text_column else 0
            has_labels = any(label in df.columns for label in ['label', 'final_test_result'])
        
            recommendations = []
        
            if task_objective.lower() == "classification":
                if not has_labels:
                    return ["❌ Classification task requires a column named 'label' or 'final_test_result' with target classes."]
        
                # Estimate number of unique classes if possible
                label_col = 'label' if 'label' in df.columns else 'final_test_result'
                num_classes = df[label_col].nunique()
        
                # Model recommendations by dataset size
                if dataset_size < 1000:
                    recommendations.append("✅ **DistilBERT** - Lightweight, fast, ideal for small-scale binary or multiclass tasks.")
                    if num_classes > 2:
                        recommendations.append("⚠️ Consider **MiniLM** for efficient multiclass classification.")
                elif dataset_size < 10000:
                    recommendations.append("✅ **RoBERTa-base** - Strong performance for mid-sized classification tasks.")
                    recommendations.append("✅ **ALBERT** - Memory-efficient and useful when hardware is limited.")
                else:
                    recommendations.append("✅ **DeBERTa-v3-base** - Excellent for large, high-performance classification.")
                    if avg_text_length > 512:
                        recommendations.append("✅ **Longformer** - Required for long sequences (input > 512 tokens).")
        
            elif task_objective.lower() == "generation":
                if not text_column:
                    return ["❌ Text generation requires a column named 'text' containing input text data."]
        
                if avg_text_length > 500:
                    recommendations.append("✅ **T5 (t5-base or t5-large)** - Good for long text-to-text generation tasks.")
                    recommendations.append("✅ **BART-large** - Strong in summarization and generation with longer inputs.")
                else:
                    recommendations.append("✅ **GPT-2 / GPT-Neo** - Suitable for short to medium-length text generation.")
                    recommendations.append("✅ **T5 (t5-small)** - Lightweight text-to-text model for short generation tasks.")
        
                if dataset_size > 10000:
                    recommendations.append("⚡ **LLaMA (via HF Transformers or API)** - High performance for large-scale generation tasks.")
                    recommendations.append("⚡ **Mistral / Mixtral** - Modern open LLMs with better efficiency and accuracy.")
        
            else:
                return ["❌ Unknown task objective. Please specify either 'Classification' or 'Generation'."]
        
            return recommendations if recommendations else ["⚠️ No model suggestions found. Please check your data and objective."]

    # Button for Model Recommendations (Bottom Left)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing
    # col1, col2 = st.columns([1, 3])
    # with col1:
    if st.button("Get LLM Recommendations"):
        recommended_models = recommend_models(df, task_objective)
            # Center the recommendations in a single line
        st.markdown(
                "<div style='text-align: center;'><strong>Recommended LLMs for Fine-Tuning:</strong> " + ", ".join(recommended_models) + "</div>",
                unsafe_allow_html=True
            )
