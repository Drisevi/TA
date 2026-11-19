import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Kunci Seed agar hasil di dashboard konsisten setiap kali di-refresh
np.random.seed(42)
tf.random.set_seed(42)

@st.cache_data
def load_and_clean_data(file):
    df_raw = pd.read_excel(file, sheet_name='Rekap')
    df_raw['Invoice Date'] = pd.to_datetime(df_raw['Invoice Date'], errors='coerce')
    df_raw['Quantity'] = pd.to_numeric(df_raw['Quantity'], errors='coerce')
    
    df_clean = df_raw.drop_duplicates().copy()
    full_weeks = pd.date_range(start=df_clean['Invoice Date'].min(), end=df_clean['Invoice Date'].max(), freq='W')
    
    weekly_pivot = (
        df_clean.groupby([pd.Grouper(key='Invoice Date', freq='W'), 'Item Description'])['Quantity']
        .sum()
        .unstack(fill_value=0.0)
        .reindex(full_weeks, fill_value=0.0)
    )
    return weekly_pivot

@st.cache_data
def classify_skus(weekly_pivot):
    train_df = weekly_pivot.iloc[2:111] # 109 minggu latih
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
            
        records.append({'SKU': sku, 'Pattern': pat, 'ADI': round(adi, 4) if pd.notnull(adi) else None, 'CV2': round(cv2, 4) if pd.notnull(cv2) else None})
        
    df_res = pd.DataFrame(records).dropna(subset=['ADI', 'CV2']).reset_index(drop=True)
    df_res.index = np.arange(1, len(df_res) + 1)
    return df_res

def make_features(series):
    df_f = pd.DataFrame({'y': series.values})
    df_f['Lag_1'] = df_f['y'].shift(1)
    df_f['Lag_2'] = df_f['y'].shift(2)
    df_f['Lag_4'] = df_f['y'].shift(4)
    # Gunakan min_periods=1 agar minggu awal tidak bernilai kosong (NaN)
    df_f['Roll_Mean_4'] = df_f['y'].shift(1).rolling(4, min_periods=1).mean()
    df_f['Roll_Std_4'] = df_f['y'].shift(1).rolling(4, min_periods=1).std().fillna(0.0)
    
    df_eff = df_f.iloc[2:].fillna(0.0)
    return df_eff.drop(columns=['y']).values, df_eff['y'].values

@st.cache_resource
def train_and_evaluate_model(sku_name, weekly_pivot):
    X, y = make_features(weekly_pivot[sku_name])
    X_train, y_train = X[:109], y[:109]
    X_test, y_test = X[109:], y[109:]
    
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = {'RF': [], 'SVR': [], 'LSTM': []}
    oof_y = []
    
    # Validasi Temporal untuk mencari Error Asli (Bebas Data Leakage)
    for tr_idx, val_idx in tscv.split(X_train):
        X_t, y_t = X_train[tr_idx], y_train[tr_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]
        
        sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
        X_tr_s = sc_X.fit_transform(X_t)
        X_val_s = sc_X.transform(X_v)
        y_tr_s = sc_y.fit_transform(y_t.reshape(-1, 1)).ravel()
        
        # Bobot resensi (recent data lebih berbobot)
        sw = np.exp(np.linspace(-1.8, 0, len(y_t)))
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
        rf.fit(X_t, y_t, sample_weight=sw)
        oof_preds['RF'].extend(np.maximum(0.0, rf.predict(X_v)))
        
        svr = SVR(kernel='rbf', C=1.5, gamma='scale')
        svr.fit(X_tr_s, y_tr_s, sample_weight=sw)
        oof_preds['SVR'].extend(np.maximum(0.0, sc_y.inverse_transform(svr.predict(X_val_s).reshape(-1, 1)).ravel()))
        
        model_lstm = Sequential([
            LSTM(32, activation='relu', input_shape=(X_tr_s.shape[1], 1)),
            Dropout(0.2), Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_tr_s.reshape(-1, X_tr_s.shape[1], 1), y_tr_s, epochs=30, verbose=0, shuffle=False)
        oof_preds['LSTM'].extend(np.maximum(0.0, sc_y.inverse_transform(model_lstm.predict(X_val_s.reshape(-1, X_val_s.shape[1], 1), verbose=0)).ravel()))
        
        oof_y.extend(y_v)
        
    oof_y = np.array(oof_y)
    
    # Hitung bobot ensemble berdasarkan error validasi temporal
    mae_val_rf = np.mean(np.abs(oof_y - np.array(oof_preds['RF']))) + 1e-6
    mae_val_svr = np.mean(np.abs(oof_y - np.array(oof_preds['SVR']))) + 1e-6
    mae_val_lstm = np.mean(np.abs(oof_y - np.array(oof_preds['LSTM']))) + 1e-6
    
    inv_sum = (1/mae_val_rf) + (1/mae_val_svr) + (1/mae_val_lstm)
    w_rf = (1/mae_val_rf) / inv_sum
    w_svr = (1/mae_val_svr) / inv_sum
    w_lstm = (1/mae_val_lstm) / inv_sum
    
    # Hitung sigma_e dari Ensemble untuk Safety Stock (100% valid, tidak pakai data test)
    val_ens = w_rf*np.array(oof_preds['RF']) + w_svr*np.array(oof_preds['SVR']) + w_lstm*np.array(oof_preds['LSTM'])
    sigma_e_val = np.std(oof_y - val_ens, ddof=1)
    
    # Prediksi aktual ke masa depan
    sc_X_full, sc_y_full = MinMaxScaler(), MinMaxScaler()
    X_train_s = sc_X_full.fit_transform(X_train)
    X_test_s = sc_X_full.transform(X_test)
    y_train_s = sc_y_full.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    sw_full = np.exp(np.linspace(-1.8, 0, len(y_train)))
    
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42).fit(X_train, y_train, sample_weight=sw_full)
    p_rf_test = np.maximum(0.0, rf_full.predict(X_test))
    
    svr_full = SVR(kernel='rbf', C=1.5, gamma='scale').fit(X_train_s, y_train_s, sample_weight=sw_full)
    p_svr_test = np.maximum(0.0, sc_y_full.inverse_transform(svr_full.predict(X_test_s).reshape(-1, 1)).ravel())
    
    lstm_full = Sequential([LSTM(32, activation='relu', input_shape=(X_train_s.shape[1], 1)), Dropout(0.2), Dense(1)])
    lstm_full.compile(optimizer='adam', loss='mse')
    lstm_full.fit(X_train_s.reshape(-1, X_train_s.shape[1], 1), y_train_s, epochs=40, verbose=0, shuffle=False)
    p_lstm_test = np.maximum(0.0, sc_y_full.inverse_transform(lstm_full.predict(X_test_s.reshape(-1, X_test_s.shape[1], 1), verbose=0)).ravel())
    
    p_ens_test = w_rf * p_rf_test + w_svr * p_svr_test + w_lstm * p_lstm_test
    
    return p_ens_test, y_test, sigma_e_val

def simulate_inventory(p_ens_test, y_test, sigma_e_val, lead_time=2.0, z_score=1.65):
    ss = z_score * sigma_e_val * np.sqrt(lead_time)
    d_hat_weekly = np.mean(p_ens_test)
    rop = (d_hat_weekly * lead_time) + ss
    
    # Hitung fill rate / stockout secara realistis berdasarkan proyeksi
    stockouts = sum(y_test > (d_hat_weekly + ss))
    fill_rate = ((len(y_test) - stockouts) / len(y_test)) * 100
    
    return {
        'Sigma_e_Val': round(sigma_e_val, 2),
        'Safety_Stock': round(ss, 2),
        'Reorder_Point': round(rop, 2),
        'Stockouts': stockouts,
        'Fill_Rate': round(fill_rate, 2)
    }

# ==========================================
# ANTARMUKA (FRONT-END) STREAMLIT
# ==========================================
if __name__ == "__main__":
    st.set_page_config(page_title="Dashboard Forecasting & Inventory", layout="wide")
    
    st.title("Sistem Simulasi Peramalan Permintaan dan Persediaan Gudang")
    st.markdown("Dashboard ini menampilkan hasil klasifikasi multi-SKU dan perhitungan simulasi Safety Stock berdasar Validasi Temporal.")
    
    uploaded_file = st.file_uploader("Unggah File Data Penjualan Gudang (.xls / .xlsx)", type=['xls', 'xlsx'])
    
    if uploaded_file is not None:
        with st.spinner('Memuat dan membersihkan data penjualan...'):
            weekly_pivot = load_and_clean_data(uploaded_file)
            df_classes = classify_skus(weekly_pivot)
            
        st.success(f"Data berhasil dimuat! Ditemukan {len(df_classes)} SKU Aktif.")
        st.subheader(f"Distribusi Pola Permintaan {len(df_classes)} SKU")
        st.dataframe(df_classes, use_container_width=True, height=450)
        
        st.subheader("Simulasi Safety Stock dan Reorder Point")
        opsi_sku = list(df_classes['SKU'].unique())
        if "NARAXONE KEMASAN 1 LITER" in opsi_sku:
            opsi_sku.insert(0, opsi_sku.pop(opsi_sku.index("NARAXONE KEMASAN 1 LITER")))
            
        sku_target = st.selectbox("Pilih Produk (SKU):", opsi_sku)
        
        if st.button("Jalankan Evaluasi Model"):
            with st.spinner(f'Menghitung Validasi Temporal untuk {sku_target}...'):
                p_ens_test, y_test, sigma_e = train_and_evaluate_model(sku_target, weekly_pivot)
                inv_decision = simulate_inventory(p_ens_test, y_test, sigma_e)
                
            st.write(f"### Hasil Evaluasi Operasional untuk: **{sku_target}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Std. Error Validasi (\u03C3e)", value=f"{inv_decision['Sigma_e_Val']:,.2f} unit".replace(',', '.'))
            col2.metric(label="Rekomendasi Safety Stock", value=f"{inv_decision['Safety_Stock']:,.2f} unit".replace(',', '.'))
            col3.metric(label="Reorder Point (ROP)", value=f"{inv_decision['Reorder_Point']:,.2f} unit".replace(',', '.'))
            
            st.info(f"**Simulasi Aliran Persediaan:** Estimasi model ini menghasilkan Fill Rate sebesar **{inv_decision['Fill_Rate']}%** (terjadi {inv_decision['Stockouts']} minggu *stockout* dari 28 minggu pengujian).")
    else:
        st.info("Silakan unggah file Excel data penjualan (misalnya: Data_Penjualan_SUN_upload.xls) untuk memulai simulasi.")
