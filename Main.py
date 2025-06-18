from Preprocess import load_data, explore_data, clean_data
from Train import train_model
from Predict import predict_new
import pandas as pd

if __name__ == "__main__":
    df = load_data()
    explore_data(df)

    print("\n🔧 Veriyi temizliyoruz...")
    X, y = clean_data(df)
    print("✅ Özellikler:\n", X.head())
    print("\n🎯 Hedef:\n", y.head())
    train_model()

    sample_data = pd.DataFrame({
        "Age": [60],
        "Gender": ["Male"],
        "Country_Region": ["Germany"],
        "Genetic_Risk": [0.6],
        "Air_Pollution": [0.4],
        "Alcohol_Use": [0.3],
        "Smoking": [0.7],
        "Obesity_Level": [0.5],
        "Cancer_Type": ["Lung"],
        "Cancer_Stage": ["Stage IV"],
        "Treatment_Cost_USD": [30000],
        "Survival_Years": [1.2],
        "Target_Severity_Score": [0]  # dummy değer (zorunlu çünkü clean_data target'lı bekliyor)
    })

    print("\n🔮 Tahmin edilen ciddiyet skoru:")
    print(predict_new(sample_data))
