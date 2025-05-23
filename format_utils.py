import pandas as pd

SYMPTOM_COLS = ['fever', 'cough', 'breathlessness', 'body_ache',
       'vomiting', 'sore_throat', 'diarrhoea', 'sputum', 'nausea',
       'nasal_discharge', 'loss_of_taste', 'loss_of_smell', 'abdominal_pain',
       'chest_pain', 'haemoptsis', 'head_ache', 'body_pain', 'weak_ness',
       'cold']  # Add your columns here

def convert_row_to_text(row):
    age = f"A {row['age']}-year-old patient"
    comorbidity = row['underlying_medical_condition']
    comorbidity_text = f"with {comorbidity}" if comorbidity and comorbidity.lower() != "none" else "with no known comorbidities"
    symptoms = [col.replace('_', ' ') for col in SYMPTOM_COLS if row.get(col) == 1]
    symptom_text = f"presented with symptoms including {', '.join(symptoms)}" if symptoms else "presented with no significant symptoms"
    gene_text = f"The RT-PCR test reported gene values as follows: E gene – {row['ct_value_screening']}, N gene – {row['ct_value_orf1b']}, and RdRp gene – {row['ct_value_rdrp']}."
    diagnosis = f"The diagnosis was {row['final_test_result']}."
    return f"{age} {comorbidity_text} {symptom_text}. {gene_text} {diagnosis}"
