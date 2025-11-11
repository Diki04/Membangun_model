import pandas as pd
import mlflow
import mlflow.sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

# Setup MLflow (Lokal + Autolog)

# Mengatur URI pelacakan ke server LOKAL
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Mengatur nama eksperimen
mlflow.set_experiment("Latihan Gym Lokal (Autolog)")

# MENGAKTIFKAN AUTOLOG
mlflow.sklearn.autolog()

print("✅ MLflow Autolog diaktifkan.")
print(f"   Tracking URI diatur ke: {mlflow.get_tracking_uri()}")
print(f"   Eksperimen diatur ke: 'Latihan Gym Lokal (Autolog)'")

# Load & Preprocess Data


def load_and_split_data(data_path, target_col="Calories_Burned"):
    """
    Memuat data dan melakukan preprocessing One-Hot Encoding
    yang kita temukan sebelumnya.
    """
    print(f"\n📥 Memuat dataset dari: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di {data_path}")
        print("Pastikan path data Anda sudah benar.")
        return None, None, None, None

    # One-Hot Encoding 
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print(f"   Meng-encode kolom: {list(categorical_cols)}")
        df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_processed = df
    
    print(f"✅ Data siap! Bentuk: {df_processed.shape}")
    
    # Memisahkan fitur (X) dan target (y)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Membagi data train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Data train: {len(X_train)} | Data test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# Jalankan Script Utama

# Definisikan path data Anda
DATA_PATH = "../Membangun_model/gym_preprocessing/gym_cleaned.csv"

if __name__ == "__main__":
    print("\n🚀 Memulai proses training...")
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH)
    
    if X_train is not None:
        # Memulai 'run' MLflow.
        # Autolog akan mencatat semua yang terjadi di dalam blok ini.
        with mlflow.start_run(run_name="Simple_RF_Autolog") as run:
            
            print(f"   Memulai Run ID: {run.info.run_id}")
            
            # Inisialisasi model
            # Autolog akan otomatis mencatat parameter ini (n_estimators, random_state)
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            
            print("   Fitting model ke data training...")
            rf_model.fit(X_train, y_train)
            
            print("   Model selesai di-fit.")
          
            # juga mencatat metrik_test
            print("   Mengevaluasi model pada data test...")
            test_score = rf_model.score(X_test, y_test)
            
            print(f"\n📈 R2 Score (Test Set): {test_score:.4f}")
            print("\n✅ Training Selesai.")
            print("Semua parameter, metrik, dan model sudah dicatat secara otomatis.")
            print(f"👉 Cek hasilnya di browser Anda: http://127.0.0.1:5000")

    else:
        print("Training dibatalkan karena data tidak ditemukan.")