import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path= "data/global_cancer_patients_2015_2024.csv"):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print("🧾 Veri seti şekli:", df.shape)
    print("\n🧩 İlk 5 satır:\n", df.head())
    print("\n🧼 Eksik veriler:\n", df.isnull().sum())
    print("\n🧠 Sütun türleri:\n", df.dtypes)

def clean_data(df):
    # Target & Features
    features = [
        'Age', 'Gender', 'Country_Region', 'Genetic_Risk', 'Air_Pollution',
        'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type',
        'Cancer_Stage', 'Treatment_Cost_USD', 'Survival_Years'
    ]
    target = 'Target_Severity_Score'

    # Drop null columns
    df = df[features + [target]].dropna()

    # Encoding
    label_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
    label_encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # İleride ters çevirmek için saklanabilir

    X = df[features]
    y = df[target]

    return X, y