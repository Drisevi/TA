import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Set seed agar hasil selalu konsisten saat didemokan
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. FUNGSI LOAD & CLEANING DATA
# ==========================================
def load_and_clean_data(filepath):
    df_raw = pd.read_excel(filepath, sheet_name='Rekap')
    df_raw['Invoice Date'] = pd.to_datetime(df_raw['Invoice Date'], errors='coerce')
    df_raw['Quantity'] = pd.to_numeric(df_raw['Quantity'], errors='coerce')
    
    # Drop duplikat eksak
    df_clean = df_raw.drop_duplicates().copy()
    
    # Agregasi kalender mingguan penuh (139 minggu)
    full_weeks = pd.date_range(start=df_clean['Invoice Date'].min(), end=df_clean['Invoice Date'].max(), freq='W')
    weekly_pivot = (
        df_clean.groupby([pd.Grouper(key='Invoice Date', freq='W'), 'Item Description'])['Quantity']
        .sum()
        .unstack(fill_value=0.0)
        .reindex(full_weeks, fill_value=0.0)
    )
    return weekly_pivot

# ==========================================
# 2. FUNGSI KLASIFIKASI ADI-CV2 (MURNI DATA LATIH)
# ==========================================
def classify_skus(weekly_pivot, n_train=109):
    train_df = weekly_pivot.iloc[2:n_train+2] # 109 minggu latih (setelah potong 2 minggu Lag)
    records = []
    
    for sku in train_df.columns:
        y_tr = train_df[sku].values
        n_tr = len(y_tr)
        nz_tr = y_tr[y_tr > 0]
        n_nz_tr = len(nz_tr)
        
        adi = n_tr / n_nz_tr if n_nz_tr > 0 else np.nan
        cv2 = (np.std(nz_tr, ddof=1) / np.mean(nz_tr)) ** 2 if n_nz_tr > 1 else (0.0 if n_nz_tr == 1 else np.nan)
        
        if n_nz_tr == 0:
            pat = "Inactive"
        elif adi <= 1.32 and cv2 <= 0.49:
            pat = "Smooth"
        elif adi <= 1.32 and cv2 > 0.49:
            pat = "Erratic"
        elif adi > 1.32 and cv2 <= 0.49:
            pat = "Intermittent"
        else:
            pat = "Lumpy"
            
        records.append({'SKU': sku, 'Pattern': pat, 'ADI': adi, 'CV2': cv2})
        
    return pd.DataFrame(records).dropna(subset=['ADI', 'CV2'])

# ==========================================
# 3. FUNGSI REKAYASA FITUR (LAG + ROLLING)
# ==========================================
def make_features(series):
    df_f = pd.DataFrame({'y': series.values})
    df_f['Lag_1'] = df_f['y'].shift(1)
    df_f['Lag_2'] = df_f['y'].shift(2)
    df_f['Lag_4'] = df_f['y'].shift(4)
    df_f['Roll_Mean_4'] = df_f['y'].shift(1).rolling(4, min_periods=2).mean()
    df_f['Roll_Std_4'] = df_f['y'].shift(1).rolling(4, min_periods=2).std().fillna(0.0)
    
    # Hapus 2 baris pertama karena NaN di fitur lag
    df_eff = df_f.iloc[2:].bfill().fillna(0.0)
    X = df_eff.drop(columns=['y']).values
    y = df_eff['y'].values
    return X, y

# ==========================================
# 4. PIPELINE MODELING & ROLLING-ORIGIN CV
# ==========================================
def train_and_evaluate_model(sku_name, weekly_pivot):
    X, y = make_features(weekly_pivot[sku_name])
    X_train, y_train = X[:109], y[:109]
    X_test, y_test = X[109:], y[109:]
    
    # 5-Fold Rolling-Origin CV untuk cari bobot Validasi
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = {'RF': [], 'SVR': [], 'LSTM': []}
    oof_y, oof_X = [], []
    
    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        
        # Scaling murni di dalam fold
        sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
        X_tr_s = sc_X.fit_transform(X_tr)
        X_val_s = sc_X.transform(X_val)
        y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        
        # RF
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_tr, y_tr)
        oof_preds['RF'].extend(np.maximum(0.0, rf.predict(X_val)))
        
        # SVR
        svr = SVR(kernel='rbf', C=1.5, gamma='scale')
        svr.fit(X_tr_s, y_tr_s)
        oof_preds['SVR'].extend(np.maximum(0.0, sc_y.inverse_transform(svr.predict(X_val_s).reshape(-1, 1)).ravel()))
        
        # LSTM
        model_lstm = Sequential([
            LSTM(32, activation='relu', input_shape=(X_tr_s.shape[1], 1)),
            Dropout(0.2), Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_tr_s.reshape(-1, X_tr_s.shape[1], 1), y_tr_s, epochs=30, verbose=0, shuffle=False)
        p_lstm = np.maximum(0.0, sc_y.inverse_transform(model_lstm.predict(X_val_s.reshape(-1, X_val_s.shape[1], 1), verbose=0)).ravel())
        oof_preds['LSTM'].extend(p_lstm)
        
        oof_y.extend(y_val)
        oof_X.append(X_val)
        
    oof_y = np.array(oof_y)
    oof_X = np.vstack(oof_X)
    
    # Hitung Temporal Validation Error untuk Bobot
    mae_val_rf = np.mean(np.abs(oof_y - np.array(oof_preds['RF']))) + 1e-6
    mae_val_svr = np.mean(np.abs(oof_y - np.array(oof_preds['SVR']))) + 1e-6
    mae_val_lstm = np.mean(np.abs(oof_y - np.array(oof_preds['LSTM']))) + 1e-6
    
    inv_sum = (1/mae_val_rf) + (1/mae_val_svr) + (1/mae_val_lstm)
    w_rf = (1/mae_val_rf) / inv_sum
    w_svr = (1/mae_val_svr) / inv_sum
    w_lstm = (1/mae_val_lstm) / inv_sum
    
    # Ekstrak standar deviasi error (Sigma_e) dari ensemble validasi untuk Safety Stock!
    val_ens = w_rf*np.array(oof_preds['RF']) + w_svr*np.array(oof_preds['SVR']) + w_lstm*np.array(oof_preds['LSTM'])
    sigma_e_val = np.std(oof_y - val_ens, ddof=1)
    
    # Train Final Model (Full 109 minggu) & Test (28 minggu holdout)
    sc_X_full, sc_y_full = MinMaxScaler(), MinMaxScaler()
    X_train_s = sc_X_full.fit_transform(X_train)
    X_test_s = sc_X_full.transform(X_test)
    y_train_s = sc_y_full.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
    p_rf_test = np.maximum(0.0, rf_full.predict(X_test))
    
    svr_full = SVR(kernel='rbf', C=1.5, gamma='scale').fit(X_train_s, y_train_s)
    p_svr_test = np.maximum(0.0, sc_y_full.inverse_transform(svr_full.predict(X_test_s).reshape(-1, 1)).ravel())
    
    lstm_full = Sequential([LSTM(32, activation='relu', input_shape=(X_train_s.shape[1], 1)), Dropout(0.2), Dense(1)])
    lstm_full.compile(optimizer='adam', loss='mse')
    lstm_full.fit(X_train_s.reshape(-1, X_train_s.shape[1], 1), y_train_s, epochs=40, verbose=0, shuffle=False)
    p_lstm_test = np.maximum(0.0, sc_y_full.inverse_transform(lstm_full.predict(X_test_s.reshape(-1, X_test_s.shape[1], 1), verbose=0)).ravel())
    
    p_ens_test = w_rf * p_rf_test + w_svr * p_svr_test + w_lstm * p_lstm_test
    
    return p_ens_test, y_test, sigma_e_val

# ==========================================
# 5. SIMULASI INVENTORY (SAFETY STOCK & ROP)
# ==========================================
def simulate_inventory(p_ens_test, sigma_e_val, lead_time=2.0, z_score=1.65):
    # Safety Stock berbasis Error Latih (Bebas Data Leakage)
    ss = z_score * sigma_e_val * np.sqrt(lead_time)
    
    # Rata-rata prediksi permintaan per minggu di periode pengujian
    d_hat_weekly = np.mean(p_ens_test)
    
    # Reorder Point (D_hat selama lead time + SS)
    rop = (d_hat_weekly * lead_time) + ss
    
    return {
        'Sigma_e_Val': round(sigma_e_val, 2),
        'Safety_Stock': round(ss, 2),
        'Reorder_Point': round(rop, 2)
    }

# Contoh Eksekusi Keseluruhan Pipeline
if __name__ == "__main__":
    filepath = "Data_Penjualan_SUN_upload.xls"
    weekly_pivot = load_and_clean_data(filepath)
    df_classes = classify_skus(weekly_pivot)
    
    # Contoh pengujian 1 SKU representatif
    sku_target = "NARAXONE KEMASAN 1 LITER"
    preds, actuals, sigma_e = train_and_evaluate_model(sku_target, weekly_pivot)
    inv_decision = simulate_inventory(preds, sigma_e)
    
    print(f"Hasil Evaluasi Operasional untuk: {sku_target}")
    print(f"Safety Stock yang disarankan : {inv_decision['Safety_Stock']} unit")
    print(f"Reorder Point (ROP)          : {inv_decision['Reorder_Point']} unit")
