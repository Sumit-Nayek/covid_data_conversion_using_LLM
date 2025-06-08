# import streamlit as st
# import pandas as pd
# import numpy as np
# # import matplotlib.pyplot as plt
# from format_utils import convert_row_to_text
# import json
# import io
# import uuid

# st.set_page_config(page_title="Medical Textifier", layout="wide")
# st.title("Medical Textifier 🏥🧠")

# # Initialize session state for storing DataFrame
# if 'df' not in st.session_state:
#     st.session_state.df = None

# # Sidebar navigation
# page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning"])

# # File uploader
# uploaded_file = st.file_uploader("Upload your medical CSV file", type=["csv"])
# if uploaded_file:
#     try:
#         st.session_state.df = pd.read_csv(uploaded_file)
#         st.session_state.df.fillna('', inplace=True)
#     except Exception as e:
#         st.error(f"Error loading CSV: {e}")
#         st.stop()

# # Page 1: Transform Data
# if page == "🔄 Transform Data":
#     st.subheader("Step 1: Convert tabular data to structured clinical text")

#     if st.session_state.df is None:
#         st.warning("Please upload a CSV file to continue.")
#         st.stop()

#     df = st.session_state.df

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

#         # Update session state with transformed DataFrame
#         st.session_state.df = df

#         # Let the user choose output format
#         export_format = st.selectbox("Choose download format", ["CSV", "JSONL"])
        
#         if export_format == "CSV":
#             st.download_button(
#                 "📥 Download Converted CSV",
#                 data=df[['text', 'label']].to_csv(index=False).encode('utf-8') if 'label' in df.columns else df[['text']].to_csv(index=False).encode('utf-8'),
#                 file_name='llm_ready_dataset.csv',
#                 mime='text/csv'
#             )
        
#         elif export_format == "JSONL":
#             records = df[['text', 'label']].to_dict(orient='records') if 'label' in df.columns else df[['text']].to_dict(orient='records')
#             jsonl_data = "\n".join([json.dumps(record) for record in records])
#             st.download_button(
#                 "📥 Download converted JSONL",
#                 data=jsonl_data.encode('utf-8'),
#                 file_name='llm_ready_dataset.jsonl',
#                 mime='application/json'
#             )

# # Page 2: Prepare for Fine-Tuning
# elif page == "📊 Prepare for Fine-Tuning":
#     st.subheader("Step 2: Fine-Tuning Analysis and Variable Prioritization")

#     if st.session_state.df is None:
#         st.warning("Please upload a CSV file in the Transform Data page to continue.")
#         st.stop()

#     df = st.session_state.df

#     st.markdown("This section helps identify key features and recommend LLMs for fine-tuning based on your data and task objective.", unsafe_allow_html=True)

#     # Objective Selection (Dropdown)
#     task_objective = st.selectbox(
#         "Select your task objective",
#         ["Classification", "Text Generation"],
#         help="Choose whether your goal is classification (e.g., predicting labels) or text generation (e.g., generating clinical notes)."
#     )

#     # Feature Scoring Logic
#     def simple_variable_score(col):
#         if df[col].nunique() == 2:
#             return 9
#         elif df[col].dtype == 'object':
#             return 7
#         elif df[col].dtype in ['int64', 'float64']:
#             return 5
#         return 3

#     scores = {col: simple_variable_score(col) for col in df.columns if col != 'text'}
#     score_df = pd.DataFrame(list(scores.items()), columns=["Feature", "LLM Score"]).sort_values(by="LLM Score", ascending=False)

#     # Button for Visualizing Feature Scores
#     st.markdown("**Feature Prioritization for Fine-Tuning:**")
#     if st.button("Visualize Feature Scores"):
#             st.write("Under construction....")
#         # fig, ax = plt.subplots(figsize=(10, 6))
#         # ax.bar(score_df["Feature"], score_df["LLM Score"], color='skyblue')
#         # ax.set_xlabel("Features")
#         # ax.set_ylabel("LLM Score (0-10)")
#         # ax.set_title("Feature Prioritization for LLM Fine-Tuning")
#         # plt.xticks(rotation=45, ha='right')
#         # plt.tight_layout()

#         # # Save plot to a bytes buffer
#         # buf = io.BytesIO()
#         # plt.savefig(buf, format='png')
#         # buf.seek(0)
#         # st.image(buf, caption="Feature Score Visualization", use_column_width=True)
#         # plt.close(fig)

#     # top_features = score_df[score_df['LLM Score'] >= 2]['Feature'].tolist()
#     # st.markdown("**Recommended features for fine-tuning:**")
#     # st.markdown(", ".join(top_features) if top_features else "_No strong features identified._")

#     # st.download_button(
#     #     "📅 Download Feature Priority CSV",
#     #     data=score_df.to_csv(index=False).encode('utf-8'),
#     #     file_name='llm_variable_scores.csv',
#     #     mime='text/csv'
#     # )

#     # Model Recommendation Logic
#     def recommend_models(df, task_objective):
#             dataset_size = len(df)
#             text_column = 'text' if 'text' in df.columns else None
#             avg_text_length = df[text_column].apply(lambda x: len(str(x).split())).mean() if text_column else 0
#             has_labels = any(label in df.columns for label in ['label', 'final_test_result'])
        
#             recommendations = []
        
#             if task_objective.lower() == "classification":
#                 if not has_labels:
#                     return ["❌ Classification task requires a column named 'label' or 'final_test_result' with target classes."]
        
#                 # Estimate number of unique classes if possible
#                 label_col = 'label' if 'label' in df.columns else 'final_test_result'
#                 num_classes = df[label_col].nunique()
        
#                 # Model recommendations by dataset size
#                 if dataset_size < 1000:
#                     recommendations.append("✅ **DistilBERT** - Lightweight, fast, ideal for small-scale binary or multiclass tasks.")
#                     if num_classes > 2:
#                         recommendations.append("⚠️ Consider **MiniLM** for efficient multiclass classification.")
#                 elif dataset_size < 10000:
#                     recommendations.append("✅ **RoBERTa-base** - Strong performance for mid-sized classification tasks.")
#                     recommendations.append("✅ **ALBERT** - Memory-efficient and useful when hardware is limited.")
#                 else:
#                     recommendations.append("✅ **DeBERTa-v3-base** - Excellent for large, high-performance classification.")
#                     if avg_text_length > 512:
#                         recommendations.append("✅ **Longformer** - Required for long sequences (input > 512 tokens).")
        
#             elif task_objective.lower() == "generation":
#                 if not text_column:
#                     return ["❌ Text generation requires a column named 'text' containing input text data."]
        
#                 if avg_text_length > 500:
#                     recommendations.append("✅ **T5 (t5-base or t5-large)** - Good for long text-to-text generation tasks.")
#                     recommendations.append("✅ **BART-large** - Strong in summarization and generation with longer inputs.")
#                 else:
#                     recommendations.append("✅ **GPT-2 / GPT-Neo** - Suitable for short to medium-length text generation.")
#                     recommendations.append("✅ **T5 (t5-small)** - Lightweight text-to-text model for short generation tasks.")
        
#                 if dataset_size > 10000:
#                     recommendations.append("⚡ **LLaMA (via HF Transformers or API)** - High performance for large-scale generation tasks.")
#                     recommendations.append("⚡ **Mistral / Mixtral** - Modern open LLMs with better efficiency and accuracy.")
        
#             else:
#                 return ["❌ Unknown task objective. Please specify either 'Classification' or 'Generation'."]
        
#             return recommendations if recommendations else ["⚠️ No model suggestions found. Please check your data and objective."]

#     # Button for Model Recommendations (Bottom Left)
#     st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing
#     # col1, col2 = st.columns([1, 3])
#     # with col1:
#     if st.button("Get LLM Recommendations"):
#         recommended_models = recommend_models(df, task_objective)
#             # Center the recommendations in a single line
#         st.markdown(
#                 "<div style='text-align: center;'><strong>Recommended LLMs for Fine-Tuning:</strong> " + ", ".join(recommended_models) + "</div>",
#                 unsafe_allow_html=True
#             )
import streamlit as st
import pandas as pd
import numpy as np
from format_utils import convert_row_to_text
import json
import io
import uuid

# Set page configuration
st.set_page_config(page_title="Medical Textifier", layout="wide")
st.title("Medical Textifier 🏥🧠")

# Initialize session state for storing DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["🔄 Transform Data", "📊 Prepare for Fine-Tuning", "📚 Glossary"])

# Fine-Tuning Guide in Sidebar
with st.sidebar.expander("📖 Fine-Tuning Guide", expanded=False):
    st.markdown("""
    ### Beginner’s Guide to Fine-Tuning LLMs

    Fine-tuning is the process of adapting a pre-trained large language model (LLM) to perform better on a specific task, such as classifying medical records or generating clinical notes.

    #### Why Fine-Tune?
    - **Improved Accuracy**: Tailors the model to your specific dataset.
    - **Efficiency**: Leverages pre-trained knowledge, requiring less data than training from scratch.
    - **Customization**: Adapts the model to domain-specific language (e.g., medical terminology).

    #### Steps to Fine-Tune an LLM
    1. **Prepare Your Data**:
       - Use this app to convert tabular data into text or prepare labeled datasets.
       - Ensure data is clean, with relevant features and labels (for classification).
    2. **Choose a Model**:
       - Select a model based on your task (see recommendations in the Prepare for Fine-Tuning page).
       - Example models:
         - **DistilBERT**: Lightweight, great for small datasets and classification.
         - **RoBERTa**: Robust for mid-sized classification tasks.
         - **T5**: Versatile for text generation tasks.
    3. **Set Up Environment**:
       - Use Python with libraries like Hugging Face Transformers, PyTorch, or TensorFlow.
       - Ensure access to a GPU for faster training (optional but recommended).
    4. **Fine-Tune the Model**:
       - Load the pre-trained model using Hugging Face’s `transformers` library.
       - Train on your dataset, adjusting hyperparameters (e.g., learning rate, epochs).
       - Example: For classification, use a `Trainer` API with your labeled data.
    5. **Evaluate and Deploy**:
       - Test the model on a validation set to check performance (e.g., accuracy, F1 score).
       - Deploy the model for inference or integrate it into your application.

    #### Recommended Resources
    - **DistilBERT**:
      - [Hugging Face Tutorial](https://huggingface.co/docs/transformers/training)
      - [Documentation](https://huggingface.co/distilbert-base-uncased)
    - **RoBERTa**:
      - [Fine-Tuning Guide](https://huggingface.co/docs/transformers/model_doc/roberta)
      - [Paper](https://arxiv.org/abs/1907.11692)
    - **T5**:
      - [Text-to-Text Tutorial](https://huggingface.co/docs/transformers/model_doc/t5)
      - [Google’s T5 Explorer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
    - **General Guides**:
      - [Hugging Face Course](https://huggingface.co/course)
      - [Practical Fine-Tuning Blog](https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-hugging-faces-trainer-4b7c9876c87f)

    Start small, experiment, and iterate to achieve the best results!
    """)

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

        # Optional: Add a 'label' column if available
        if 'final_test_result' in df.columns:
            df['label'] = df['final_test_result']

        # Update session state
        st.session_state.df = df

        # Export format selection
        export_format = st.selectbox("Choose download format", ["CSV", "JSONL"])
        
        if export_format == "CSV":
            st.download_button(
                "📥 Download Converted CSV",
                data=df[['text', 'label']].to_csv(index=False).encode('utf-8') if 'label' in df.columns else df[['text']].to_csv(index=False).encode('utf-8'),
                file_name='llm_ready_dataset.csv',
                mime='text/csv'
            )
        
        elif export_format == "JSONL":
            records = df[['text', 'label']].to_dict(orient='records') if 'label' in df.columns else df[['text']].to_dict(orient='records')
            jsonl_data = "\n".join([json.dumps(record) for record in records])
            st.download_button(
                "📥 Download converted JSONL",
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

    st.markdown("This section helps identify key features, recommend LLMs for fine-tuning, and detect outliers in your data.", unsafe_allow_html=True)

    # Objective Selection
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

    # Feature Scores Visualization
    st.markdown("**Feature Prioritization for Fine-Tuning:**")
    if st.button("Visualize Feature Scores"):
        st.write("Under construction....")

    # Outlier Detection Logic
    st.markdown("**Outlier Detection:**")
    st.markdown("Identify potential data quality issues in numerical columns using Z-Score or IQR methods.")

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numerical_cols:
        st.warning("No numerical columns found in the dataset for outlier detection.")
    else:
        # Select outlier detection method
        outlier_method = st.selectbox("Select outlier detection method", ["Z-Score", "IQR"])

        def detect_outliers_zscore(series, threshold=3):
            """Detect outliers using z-scores."""
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        def detect_outliers_iqr(series):
            """Detect outliers using IQR method."""
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)

        if st.button("Detect Outliers"):
            outliers = []
            for col in numerical_cols:
                # Skip NaN values
                series = df[col].dropna()
                if series.empty:
                    continue
                
                if outlier_method == "Z-Score":
                    outlier_mask = detect_outliers_zscore(series)
                    reason = f"Z-Score > 3 (Value too far from mean)"
                else:  # IQR
                    outlier_mask = detect_outliers_iqr(series)
                    reason = f"Outside IQR bounds (Q1 - 1.5*IQR or Q3 + 1.5*IQR)"

                # Get indices of outliers
                outlier_indices = series[outlier_mask].index
                for idx in outlier_indices:
                    outliers.append({
                        "Row Index": idx,
                        "Column": col,
                        "Value": df.at[idx, col],
                        "Reason": reason
                    })

            if outliers:
                outlier_df = pd.DataFrame(outliers)
                st.success(f"Found {len(outliers)} potential outliers.")
                st.dataframe(outlier_df)

                # Download outlier report
                st.download_button(
                    "📥 Download Outlier Report",
                    data=outlier_df.to_csv(index=False).encode('utf-8'),
                    file_name='outlier_report.csv',
                    mime='text/csv'
                )
            else:
                st.info("No outliers detected in the numerical columns.")

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
        
            label_col = 'label' if 'label' in df.columns else 'final_test_result'
            num_classes = df[label_col].nunique()
        
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

    # Model Recommendations
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Get LLM Recommendations"):
        recommended_models = recommend_models(df, task_objective)
        st.markdown(
            "<div style='text-align: center;'><strong>Recommended LLMs for Fine-Tuning:</strong> " + ", ".join(recommended_models) + "</div>",
            unsafe_allow_html=True
        )

# Page 3: Glossary
elif page == "📚 Glossary":
    st.subheader("Glossary of Key Terms")

    st.markdown("""
    Below is a glossary of terms used in this application, with definitions and examples. Hover over terms marked with a dotted underline to see tooltips with brief explanations.
    """)

    # Tooltip CSS and JavaScript
    tooltip_style = """
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #f4f4f4;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    </style>
    """

    # Glossary Table with Tooltips
    glossary_table = """
    <table>
        <tr>
            <th>Term</th>
            <th>Definition</th>
            <th>Example</th>
        </tr>
        <tr>
            <td><span class="tooltip">Classification<span class="tooltiptext">A machine learning task where the model predicts a category or label for input data.</span></span></td>
            <td>A task where a model predicts a discrete label or category for input data, such as diagnosing a medical condition.</td>
            <td>Predicting whether a patient has diabetes (Yes/No) based on medical records.</td>
        </tr>
        <tr>
            <td><span class="tooltip">Text Generation<span class="tooltiptext">A task where a model produces coherent text based on input prompts.</span></span></td>
            <td>A task where a model generates coherent text, such as clinical notes or summaries, based on input data.</td>
            <td>Generating a clinical note from a patient’s symptoms and medical history.</td>
        </tr>
        <tr>
            <td><span class="tooltip">DistilBERT<span class="tooltiptext">A smaller, faster version of BERT, ideal for classification tasks with limited resources.</span></span></td>
            <td>A lightweight version of the BERT model, optimized for classification tasks with smaller datasets or limited computational resources.</td>
            <td>Fine-tuning DistilBERT to classify medical records as positive or negative for a condition.</td>
        </tr>
        <tr>
            <td><span class="tooltip">RoBERTa<span class="tooltiptext">An optimized version of BERT with better performance on classification tasks.</span></span></td>
            <td>An enhanced version of BERT, trained with more data and optimized for tasks like classification and question answering.</td>
            <td>Using RoBERTa to classify patient notes into multiple sclerosis categories.</td>
        </tr>
        <tr>
            <td><span class="tooltip">T5<span class="tooltiptext">A text-to-text model that handles tasks like generation and summarization.</span></span></td>
            <td>A text-to-text transformer model that can perform tasks like text generation, summarization, and translation by framing them as text-to-text problems.</td>
            <td>Fine-tuning T5 to generate concise clinical summaries from detailed patient records.</td>
        </tr>
        <tr>
            <td><span class="tooltip">Fine-Tuning<span class="tooltiptext">Adapting a pre-trained model to a specific task using a smaller dataset.</span></span></td>
            <td>The process of adapting a pre-trained model to a specific task by training it on a smaller, task-specific dataset.</td>
            <td>Fine-tuning a model to predict medical diagnoses from patient data.</td>
        </tr>
        <tr>
            <td><span class="tooltip">LLM<span class="tooltiptext">Large Language Model, a powerful AI model trained on vast text data.</span></span></td>
            <td>Large Language Model, a type of AI model trained on vast amounts of text to understand and generate human-like language.</td>
            <td>Models like RoBERTa or T5 used for medical text analysis.</td>
        </tr>
    </table>
    """

    # Render Glossary with Tooltips
    st.markdown(tooltip_style + glossary_table, unsafe_allow_html=True)
