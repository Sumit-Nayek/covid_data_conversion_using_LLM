# --- format_utils.py ---
import pandas as pd

SYMPTOM_COLS = [
    'fever', 'cough', 'breathlessness', 'body_ache', 'vomiting', 'sore_throat',
    'diarrhoea', 'sputum', 'nausea', 'nasal_discharge', 'loss_of_taste',
    'loss_of_smell', 'abdominal_pain', 'chest_pain', 'haemoptsis',
    'head_ache', 'body_pain', 'weak_ness', 'cold'
]
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
        age = row.get('age', 'Unknown')
        comorbidity = str(row.get('underlying_medical_condition', '')).strip()
        comorbidity_text = (
            "with a history of underlying medical conditions"
            if comorbidity == '1' else
            "with no known comorbidities"
        )

        test_result = str(row.get('final_test_result', 'Unknown')).capitalize()
        ct_screening = row.get('ct_value_screening', 'NA')
        ct_rdrp = row.get('ct_value_rdrp', 'NA')
        ct_orf1b = row.get('ct_value_orf1b', 'NA')

        symptoms = [
            symptom.replace('_', ' ') for symptom in SYMPTOM_COLS
            if str(row.get(symptom, '0')).strip() in ['1', 'yes', 'Yes']
        ]

        if symptoms:
            symptom_text = ", ".join(symptoms)
            symptoms_text = f"Current symptoms include {symptom_text}."
        else:
            symptoms_text = "No other symptoms were reported at the time of examination."

        return (
            f"**Patient Summary:**  \n"
            f"A {age}-year-old individual presents {comorbidity_text}. The patient tested **{test_result}** for COVID-19.  \n"
            f"{symptoms_text}  \n\n"
            f"**Diagnostic Findings:**  \n"
            f"- Ct Values:  \n"
            f"  - Screening Gene: **{ct_screening}**  \n"
            f"  - RdRp Gene: **{ct_rdrp}**  \n"
            f"  - ORF1b Gene: **{ct_orf1b}**  \n\n"
            f"**Clinical Impression:**  \n"
            f"The presence of {symptom_text if symptoms else 'no symptoms'} with Ct values as noted suggests the need for clinical evaluation. "
            f"Underlying conditions may increase risk and warrant close monitoring."
        )

    except Exception as e:
        print(f"Error processing row: {e}")
        return "Conversion error"
