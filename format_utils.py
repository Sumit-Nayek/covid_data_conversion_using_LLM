# import pandas as pd

# SYMPTOM_COLS = ['fever', 'cough', 'breathlessness', 'body_ache',
#        'vomiting', 'sore_throat', 'diarrhoea', 'sputum', 'nausea',
#        'nasal_discharge', 'loss_of_taste', 'loss_of_smell', 'abdominal_pain',
#        'chest_pain', 'haemoptsis', 'head_ache', 'body_pain', 'weak_ness',
#        'cold']  # Add your columns here

# def convert_row_to_text(row):
#     try:
#         # Safely get comorbidity with default value
#         comorbidity = str(row.get('underlying_medical_condition', '')).strip()
#         comorbidity_text = (f"with {comorbidity}" 
#                            if comorbidity and comorbidity.lower() != "none" 
#                            else "with no known comorbidities")
        
#         # Add other fields similarly with proper error handling
#         # ...
        
#         return f"Patient presents {comorbidity_text}."  # Your complete text here
        
#     except Exception as e:
#         print(f"Error processing row: {e}")
#         return "Conversion error"  # or handle differently

# --- format_utils.py ---
import pandas as pd

SYMPTOM_COLS = [
    'fever', 'cough', 'breathlessness', 'body_ache', 'vomiting', 'sore_throat',
    'diarrhoea', 'sputum', 'nausea', 'nasal_discharge', 'loss_of_taste',
    'loss_of_smell', 'abdominal_pain', 'chest_pain', 'haemoptsis',
    'head_ache', 'body_pain', 'weak_ness', 'cold'
]

def convert_row_to_text(row):
    try:
        comorbidity = str(row.get('underlying_medical_condition', '')).strip()
        comorbidity_text = (
            f"with {comorbidity}" if comorbidity and comorbidity.lower() != "none" 
            else "with no known comorbidities"
        )

        symptoms = [symptom.replace('_', ' ') for symptom in SYMPTOM_COLS if str(row.get(symptom, '')).strip().lower() == 'yes']
        symptoms_text = (
            "Symptoms include " + ", ".join(symptoms) + "."
            if symptoms else "No notable symptoms reported."
        )

        return f"Patient presents {comorbidity_text}. {symptoms_text}"

    except Exception as e:
        print(f"Error processing row: {e}")
        return "Conversion error"

