import streamlit as st
import pandas as pd
from format_utils import convert_row_to_text

st.title("Medical Textifier 🏥🧠")
st.write("Upload your medical dataset to convert tabular data into structured clinical text.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ['comorbidity']  # Add other required columns here
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Handle NaN values and ensure string type for text conversion
            df.fillna('', inplace=True)
            
            # Apply the conversion
            df['Generated Text'] = df.apply(convert_row_to_text, axis=1)
            
            st.success("Conversion complete!")
            st.dataframe(df[['Generated Text']])
            
            # Download button
            st.download_button(
                label="Download Text Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='textified_output.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
