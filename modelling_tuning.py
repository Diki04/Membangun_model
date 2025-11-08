import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(data_dir):
    """Memuat data training dan testing yang sudah diproses."""
    print("Memuat data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
    
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_test_df = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    
    # Ambil nilai dari kolom pertama (target) dan ubah jadi 1D array
    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values
    
    return X_train, X_test, y_train, y_test

def train_with_tuning(X_train, y_train, X_test, y_test):
    """Melatih model dengan hyperparameter tuning dan log ke DagsHub."""
    
    # Mulai MLflow Run
    # 'run_name' akan jadi nama eksperimen di DagsHub
    with mlflow.start_run(run_name="Advanced RF Tuning (Calories)") as run:
        
        # 1. Definisikan Hyperparameter Grid untuk Tuning
        param_grid = {
            'n_estimators': [50, 100],        # Kurangi jumlahnya agar tuning cepat
            'max_depth': [None, 10],
            'min_samples_leaf': [2, 4]
        }
        
        # Setup model dan GridSearchCV
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=3,  # CV 3 agar lebih cepat
            n_jobs=-1, 
            scoring='neg_mean_squared_error' # Scoring untuk regresi
        )
        
        print("Memulai Hyperparameter Tuning...")
        grid_search.fit(X_train, y_train)
        
        # Dapatkan model dan parameter terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Parameters: {best_params}")
        
        # Log Parameters
        print("Logging parameters...")
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_metric", "neg_mean_squared_error")

        # Evaluasi model
        y_pred = best_model.predict(X_test)
        
        # Hitung metrik
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Metrik tambahan 1
        
        print(f"R2 Score: {r2}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        # Log Metrics (Autolog + 2 nilai tambahan)
        print("Logging metrics...")
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        # Metrik tambahan (kriteria Advance)
        mlflow.log_metric("root_mean_squared_error", rmse) # Metrik tambahan 1
        mlflow.log_metric("best_cv_score", grid_search.best_score_) # Metrik tambahan 2

        # Buat dan Log Artifact (Plot)
        print("Logging artifact (plot)...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
        plt.xlabel("Actual Calories Burned")
        plt.ylabel("Predicted Calories Burned")
        plt.title("Actual vs. Predicted Calories")
        
        # Simpan plot sebagai file
        plot_path = "actual_vs_predicted.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Log artifact
        mlflow.log_artifact(plot_path)
        
        # Log Model
        print("Logging model...")
        mlflow.sklearn.log_model(best_model, "random-forest-regressor")
        
        print("\n=== Run Selesai ===")
        print(f"Run ID: {run.info.run_id}")
        print("Cek eksperimen di DagsHub!")

if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/dikicompanyy/Membangun_model.mlflow'

    os.environ['MLFLOW_TRACKING_USERNAME'] = 'dikicompanyy'
    
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '47f11a340d5c2142c9f6f6b3d68a9f1d913799de'
    
    DATA_DIR = "gym_preprocessing"
    
    # Muat data
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    
    # Latih model
    train_with_tuning(X_train, y_train, X_test, y_test)