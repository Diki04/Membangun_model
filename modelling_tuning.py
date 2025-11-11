import os
import shutil
import json  
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.utils import estimator_html_repr

# Setup MLflow + DagsHub
load_dotenv()

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

dagshub.init(repo_owner="dikicompanyy", repo_name="Membangun_model", mlflow=True)
print("✅ Koneksi MLflow & DagsHub berhasil diinisialisasi.")


# NYALAKAN KODE LOKAL (Buka comment untuk menjalankan lokal dan comment seluruh kode di atas yaitu kode dagshub)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Latihan Gym Lokal (Tuning Manual)") 
# print("✅ Koneksi MLflow diatur ke LOKAL (http://127.0.0.1:5000)")

# Load dataset (Termasuk One-Hot Encoding)

def load_and_split_data(data_path, target_col="Calories_Burned", test_size=0.2):
    print(f"📥 Memuat dataset tunggal dari: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset dimuat! Bentuk data awal: {df.shape}")

    # --- Mulai One-Hot Encoding ---
    print("🔧 Melakukan One-Hot Encoding untuk data kategorikal...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if not categorical_cols.empty:
        print(f"   Kolom 'object' yang ditemukan: {list(categorical_cols)}")
        df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✅ Encoding selesai! Bentuk data baru: {df_processed.shape}")
    else:
        print("   Tidak ditemukan kolom 'object' untuk di-encode.")
        df_processed = df

    if target_col not in df_processed.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset!")

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Mengambil daftar nama fitur SETELAH di-encode untuk artefak
    feature_names = X.columns.tolist() 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=42, random_state=42)
    print(f"📊 Data train: {len(X_train)} | Data test: {len(X_test)}")
    
    # Mengirimkan 'feature_names' untuk di-log sebagai artefak
    return X_train, X_test, y_train, y_test, feature_names

# Training dan Logging
def train_with_tuning(X_train, y_train, X_test, y_test, feature_names):
    """
    Melatih model, melakukan tuning, dan me-log semua metrik & artefak
    sesuai kriteria Skilled dan Advanced.
    """
    with mlflow.start_run(run_name="RF_Tuning_Advanced (Calories_Burned)") as run:
        
        # 1. Proses Tuning Model 
        param_dist = {
            'n_estimators': randint(low=100, high=1000),
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_leaf': randint(low=1, high=10),
            'min_samples_split': randint(low=2, high=20),
            'max_features': [None, 'sqrt', 'log2'] 
        }
        rf = RandomForestRegressor(random_state=42)
        print("🚀 Memulai pelatihan dan tuning (RandomizedSearchCV)...")

        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42,
            verbose=1
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"🏆 Parameter terbaik: {best_params}")

        # 2. Logging Parameter 
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_iter", 50)
        mlflow.log_param("test_size", len(X_test) / (len(X_train) + len(X_test)))

        # 3. Evaluasi Model (pada Test Set) 
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Membuat dictionary metrik untuk 'metric_info.json'
        metrics = {
            "r2_score": r2,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse
        }
        print(f"📈 R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # 4. Logging Metrik
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_score", random_search.best_score_)
        
        # Logging Artefak
        print("\n--- Memulai Logging Artefak ---")

        # Artefak #1: metric_info.json 
        print("📝 Menyimpan metrik test set ke metric_info.json...")
        metric_path = "metric_info.json"
        with open(metric_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metric_path)

        # Artefak #2: estimator.html 
        print("📝 Menyimpan visualisasi estimator ke estimator.html...")
        estimator_path = "estimator.html"
        with open(estimator_path, 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(best_model))
        mlflow.log_artifact(estimator_path)

        # Artefak #3: Plot Performa Training
        print("🖼️  Membuat plot performa data TRAINING...")
        y_pred_train = best_model.predict(X_train)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, y_pred_train, alpha=0.5, color='green', label='Training Data')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--', color='red', label='Ideal Line')
        plt.xlabel("Actual Calories Burned (Training)")
        plt.ylabel("Predicted Calories Burned (Training)")
        plt.title("Actual vs Predicted (Training Set)")
        plt.legend()
        plt.tight_layout()
        training_plot_path = "training_performance_plot.png" 
        plt.savefig(training_plot_path)
        plt.close()
        mlflow.log_artifact(training_plot_path)

        # Artefak #4: Plot Performa Test Set 
        print("🖼️  Membuat plot performa data TEST...")
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Test Data')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Ideal Line')
        plt.xlabel("Actual Calories Burned (Test)")
        plt.ylabel("Predicted Calories Burned (Test)")
        plt.title("Actual vs Predicted (Test Set)")
        plt.legend()
        plt.tight_layout()
        plot_path = "actual_vs_predicted_.png" 
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Daftar Fitur 
        print("📝 Menyimpan daftar fitur ke features.txt...")
        features_path = "features.txt"
        with open(features_path, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        mlflow.log_artifact(features_path)
        
        # Logging Model 
        print("\n--- Menyimpan Model ke DagsHub ---")
        model_dir = "rf_model"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir) 
        
        mlflow.sklearn.save_model(best_model, model_dir)
        mlflow.log_artifact(model_dir) 

        print("\n✅ Training & Logging Selesai!")
        print(f"🔗 Run ID: {run.info.run_id}")
        print("Cek hasil eksperimen di DagsHub!")

# Jalankan Script Utama

if __name__ == "__main__":
    DATA_PATH = "../Membangun_model/gym_preprocessing/gym_cleaned.csv"
    
    # Menangkap 'feature_names' dari fungsi load_data
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(DATA_PATH)
    
    # Mengirim 'feature_names' ke fungsi training
    train_with_tuning(X_train, y_train, X_test, y_test, feature_names)