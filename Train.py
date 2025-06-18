from Preprocess import load_data, clean_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt


def plot_predictions(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Preds')
    plt.title('Actual vs Preds Target Severity Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_model():
    df = load_data()
    X, y = clean_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid Search -> Hyper param tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    # Best model selection
    best_model = grid_search.best_estimator_
    print("✅ En iyi parametreler:", grid_search.best_params_)

    # Predictions
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"📊 MAE: {mae:.2f}")
    print(f"📉 RMSE: {rmse:.2f}")

    # Write Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/severity_model.pkl")
    print("✅ Fine-tuned model kaydedildi.")

    # Figure
    plot_predictions(y_test, predictions)

