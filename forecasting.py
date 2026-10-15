import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Mengimpor library Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS KUSTOM
# ==========================================
st.set_page_config(page_title="Demand Forecasting AI", page_icon="📈", layout="wide")

# CSS Kustom untuk mempercantik tampilan
st.markdown("""
    <style>
    .main-title {font-size: 38px; font-weight: 800; color: #1E3A8A; text-align: center; margin-bottom: -15px;}
    .sub-title {font-size: 18px; color: #64748B; text-align: center; margin-bottom: 30px;}
    .st-tabs {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Aplikasi <i>Demand Forecasting</i> (AI)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Cerdas Pengelolaan Stok PT Samudra Utama Narapati</p>', unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. FUNGSI PENDUKUNG
# ==========================================
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    df_grouped = df.groupby(['Invoice Date', 'Item Description'])['Quantity'].sum().reset_index()
    return df_grouped

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 2)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 3. PANEL SAMPING (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=80)
st.sidebar.markdown("## ⚙️ Panel Kendali")

uploaded_file = st.sidebar.file_uploader("📂 1. Upload Data Penjualan (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Memuat data
    df_all = load_and_preprocess_data(uploaded_file)
    
    st.sidebar.markdown("---")
    item_list = df_all['Item Description'].unique()
    selected_item = st.sidebar.selectbox("📦 2. Pilih Produk:", item_list)

    # Filter & Resample Data
    df_item = df_all[df_all['Item Description'] == selected_item].copy()
    df_item.set_index('Invoice Date', inplace=True)
    df_weekly = df_item['Quantity'].resample('W').sum().fillna(0).reset_index()

    st.sidebar.markdown("---")
    model_choice = st.sidebar.radio("🤖 3. Pilih Algoritma AI:", [
        "Random Forest", 
        "Support Vector Machine (SVM)",
        "Long Short-Term Memory (LSTM)",
        "Hybrid Model (LSTM + Random Forest)",
        "Voting Regressor (Rata-rata 3 Model)"
    ])
    
    jalankan = st.sidebar.button("🚀 Jalankan Analisis AI", use_container_width=True)

    # ==========================================
    # 4. AREA UTAMA (TABS)
    # ==========================================
    tab1, tab2 = st.tabs(["📊 Eksplorasi Data", "🔮 Prediksi & Evaluasi Model"])

    # ----- TAB 1: EKSPLORASI DATA -----
    with tab1:
        st.markdown(f"### 📋 Riwayat Permintaan Mingguan")
        st.info(f"Menampilkan tren historis untuk produk: **{selected_item}**")
        
        # Desain Grafik Historis
        fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
        ax_hist.plot(df_weekly['Invoice Date'], df_weekly['Quantity'], color='#2563EB', linewidth=2, marker='o', markersize=4)
        ax_hist.set_title("Fluktuasi Permintaan Pasar", fontsize=12, fontweight='bold', color='#334155')
        ax_hist.set_ylabel("Kuantitas (Unit)")
        ax_hist.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_hist)
        
        with st.expander("Tampilkan Tabel Data Mentah"):
            st.dataframe(df_weekly, use_container_width=True)

    # ----- TAB 2: PREDIKSI & EVALUASI -----
    with tab2:
        if not jalankan:
            st.warning("👈 Silakan klik tombol **'Jalankan Analisis AI'** di menu samping untuk memproses model.")
        else:
            with st.spinner(f"Memproses {model_choice}... Algoritma sedang mempelajari pola data."):
                
                # Persiapan Data (Lag Features)
                df_weekly['Lag_1'] = df_weekly['Quantity'].shift(1)
                df_weekly['Lag_2'] = df_weekly['Quantity'].shift(2)
                df_weekly.dropna(inplace=True)

                X = df_weekly[['Lag_1', 'Lag_2']]
                y = df_weekly['Quantity']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Preprocessing LSTM
                scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

                # Inisialisasi
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                svm_model = SVR(kernel='rbf')
                y_pred = None

                # Eksekusi Model
                if model_choice == "Random Forest":
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)

                elif model_choice == "Support Vector Machine (SVM)":
                    svm_model.fit(X_train, y_train)
                    y_pred = svm_model.predict(X_test)

                elif model_choice == "Long Short-Term Memory (LSTM)":
                    lstm = build_lstm_model()
                    lstm.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                    y_pred = scaler_y.inverse_transform(lstm.predict(X_test_lstm)).flatten()

                elif model_choice == "Hybrid Model (LSTM + Random Forest)":
                    lstm = build_lstm_model()
                    lstm.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                    train_pred = scaler_y.inverse_transform(lstm.predict(X_train_lstm)).flatten()
                    residu_train = y_train.values - train_pred
                    rf_model.fit(X_train, residu_train)
                    test_pred_lstm = scaler_y.inverse_transform(lstm.predict(X_test_lstm)).flatten()
                    test_pred_residu = rf_model.predict(X_test)
                    y_pred = test_pred_lstm + test_pred_residu

                elif model_choice == "Voting Regressor (Rata-rata 3 Model)":
                    rf_model.fit(X_train, y_train)
                    pred_rf = rf_model.predict(X_test)
                    svm_model.fit(X_train, y_train)
                    pred_svm = svm_model.predict(X_test)
                    lstm = build_lstm_model()
                    lstm.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                    pred_lstm = scaler_y.inverse_transform(lstm.predict(X_test_lstm)).flatten()
                    y_pred = (pred_rf + pred_svm + pred_lstm) / 3

                y_pred = np.maximum(y_pred, 0) # Cegah nilai minus

                # Kalkulasi Metrik
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if sum(mask) > 0 else 0

                # Tampilan Hasil Evaluasi
                st.success(f"✅ Analisis selesai! Menampilkan hasil evaluasi untuk model: **{model_choice}**")
                
                # Kartu Metrik
                st.markdown("### 🎯 Kinerja Model (Tingkat Error)")
                col1, col2, col3 = st.columns(3)
                col1.metric("📉 MAE (Mean Absolute Error)", round(mae, 2))
                col2.metric("📊 RMSE (Root Mean Sq Error)", round(rmse, 2))
                col3.metric("🎯 MAPE (Persentase Error)", f"{round(mape, 2)}%")

                st.divider()

                # Grafik Prediksi vs Aktual
                st.markdown("### 📈 Komparasi Data Aktual vs Prediksi AI")
                fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
                ax_pred.plot(df_weekly['Invoice Date'].iloc[-len(y_test):], y_test.values, label='Kenyataan (Data Aktual)', color='#0EA5E9', linewidth=2, marker='o')
                ax_pred.plot(df_weekly['Invoice Date'].iloc[-len(y_test):], y_pred, label='Tebakan AI (Prediksi)', color='#E11D48', linewidth=2, linestyle='--', marker='x')
                ax_pred.set_ylabel("Kuantitas (Unit)")
                ax_pred.grid(True, linestyle=':', alpha=0.7)
                ax_pred.legend(loc="upper right", frameon=True, shadow=True)
                
                st.pyplot(fig_pred)

else:
    st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: #F8FAFC; border-radius: 10px; border: 1px dashed #CBD5E1;'>
            <h3 style='color: #475569;'>Aplikasi Menunggu Data 📂</h3>
            <p style='color: #64748B;'>Silakan unggah file <b>Data_Penjualan_Bersih.xlsx</b> Anda pada panel di sebelah kiri untuk memulai proses peramalan (forecasting).</p>
        </div>
    """, unsafe_allow_html=True)
