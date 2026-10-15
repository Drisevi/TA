import subprocess
import sys

# Otomatis instal xlrd dan openpyxl di server Streamlit Cloud jika belum ada
for pkg in ["xlrd", "openpyxl"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. KONFIGURASI HALAMAN & DESAIN CSS
# ==========================================
st.set_page_config(page_title="Demand Forecasting AI - PT SUN", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    .main-title {font-size: 34px; font-weight: 800; color: #1E3A8A; text-align: center; margin-bottom: 0px;}
    .sub-title {font-size: 16px; color: #475569; text-align: center; margin-bottom: 20px;}
    .step-badge {background-color: #DBEAFE; color: #1E40AF; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 14px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📦 Sistem Pengelolaan Stok Gudang & Demand Forecasting Berbasis AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Studi Kasus: PT Samudra Utama Narapati | Metode: RF, SVM, LSTM, Hybrid & Ensemble</p>', unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. FUNGSI PEMROSESAN & PEMODELAN
# ==========================================
@st.cache_data
def load_and_clean_data(uploaded_file):
    df_raw = pd.read_excel(uploaded_file)
    total_raw = len(df_raw)
    
    # 1. Data Cleaning: Hapus duplikat & nilai kosong
    missing_count = int(df_raw.isnull().sum().sum())
    df_clean = df_raw.dropna().copy()
    dup_count = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates().copy()
    
    # Pastikan tipe data benar
    df_clean['Invoice Date'] = pd.to_datetime(df_clean['Invoice Date'])
    df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce').fillna(0)
    df_clean = df_clean[df_clean['Quantity'] >= 0]
    
    df_grouped = df_clean.groupby(['Invoice Date', 'Item Description'])['Quantity'].sum().reset_index()
    
    cleaning_stats = {
        "total_raw": total_raw,
        "missing_removed": missing_count,
        "duplicates_removed": dup_count,
        "total_clean": len(df_clean)
    }
    return df_grouped, cleaning_stats

def hitung_metrik(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else 0.0
    return mae, rmse, mape

def build_lstm_model(units=50, dropout_rate=0.2, input_dim=2):
    tf.random.set_seed(42)
    np.random.seed(42)
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(1, input_dim)),
        Dropout(dropout_rate), # Mitigasi overfitting sesuai Bab 3.2.1
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 3. SIDEBAR (PANEL KENDALI)
# ==========================================
st.sidebar.markdown("## ⚙️ Panel Kendali Sistem")
uploaded_file = st.sidebar.file_uploader("📂 1. Upload Data Penjualan Historis", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_all, clean_stats = load_and_clean_data(uploaded_file)
    st.sidebar.success("✅ Data berhasil dimuat & dibersihkan!")

    st.sidebar.markdown("---")
    item_list = sorted(df_all['Item Description'].unique())
    selected_item = st.sidebar.selectbox("📦 2. Pilih Produk Gudang:", item_list)

    # Filter produk dan transformasi ke mingguan
    df_item = df_all[df_all['Item Description'] == selected_item].copy()
    df_item.set_index('Invoice Date', inplace=True)
    df_weekly = df_item['Quantity'].resample('W').sum().fillna(0).reset_index()

    # Buat fitur Time-Series (Lag 1, Lag 2, Moving Average Konvensional)
    df_weekly['Lag_1'] = df_weekly['Quantity'].shift(1)
    df_weekly['Lag_2'] = df_weekly['Quantity'].shift(2)
    df_weekly['MA_Konvensional'] = df_weekly[['Lag_1', 'Lag_2']].mean(axis=1)
    df_model = df_weekly.dropna().reset_index(drop=True)

    st.sidebar.markdown("---")
    model_choice = st.sidebar.radio("🤖 3. Pilih Model Fokus (Sesuai Bab III):", [
        "Random Forest (RF)",
        "Support Vector Machine (SVM)",
        "Long Short-Term Memory (LSTM)",
        "Hybrid Model (LSTM + RF)",
        "Voting Regressor (Rata-rata 3 Model)",
        "Weighted Average (Bobot Performa)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 4. Pengaturan Tuning & Prediksi")
    rf_trees_tune = st.sidebar.slider("Jumlah Pohon RF (Tuning):", 50, 300, 150, step=50)
    svm_c_tune = st.sidebar.slider("Regularisasi L2 SVM (Parameter C):", 1.0, 100.0, 10.0)
    lstm_dropout = st.sidebar.slider("Dropout Rate LSTM (Cegah Overfitting):", 0.1, 0.5, 0.2, step=0.1)
    horizon_weeks = st.sidebar.slider("Horizon Prediksi Stok ke Depan (Minggu):", 4, 12, 4)

    jalankan = st.sidebar.button("🚀 Jalankan Seluruh Alur Model", use_container_width=True)

    # Pembagian Data Latih (80%) dan Data Uji (20%) sesuai Bab 3.2.1
    X = df_model[['Lag_1', 'Lag_2']]
    y = df_model['Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    test_dates = df_model['Invoice Date'].iloc[-len(y_test):]

    # ==========================================
    # 4. EMPAT TAB SESUAI GAMBAR 3.1 PROPOSAL
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Pemrosesan Data & EDA",
        "2️⃣ Latih & Evaluasi Model (+ K-Fold)",
        "3️⃣ Tuning Hiperparameter",
        "4️⃣ Implementasi Model (Stok Masa Depan)"
    ])

    # ------------------------------------------
    # TAB 1: PEMROSESAN DATA (ALUR KIRI GAMBAR 3.1)
    # ------------------------------------------
    with tab1:
        st.markdown('<span class="step-badge">Tahap 1 (Gambar 3.1): Peroleh Data ➔ Praproses Data ➔ Bagi Data Latih & Uji (80:20)</span>', unsafe_allow_html=True)
        st.markdown(f"### 📊 Hasil Akuisisi dan Pengolahan Data: **{selected_item}**")

        # Ringkasan Data Cleaning
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transaksi Mentah", f"{clean_stats['total_raw']:,} baris")
        c2.metric("Data Kosong Dihapus", f"{clean_stats['missing_removed']} baris")
        c3.metric("Data Ganda Dihapus", f"{clean_stats['duplicates_removed']} baris")
        c4.metric("Total Minggu (Time-Series)", f"{len(df_model)} minggu")

        st.divider()

        # Exploratory Data Analysis (EDA) & Visualisasi Pembagian 80:20
        st.markdown("#### 📈 Exploratory Data Analysis (EDA) & Pembagian Data (80% Latih : 20% Uji)")
        fig_split, ax_split = plt.subplots(figsize=(12, 4))
        train_dates = df_model['Invoice Date'].iloc[:len(y_train)]
        ax_split.plot(train_dates, y_train, label='Data Latih / Training (80%)', color='#2563EB', linewidth=2)
        ax_split.plot(test_dates, y_test, label='Data Uji / Testing (20%)', color='#F59E0B', linewidth=2)
        ax_split.axvline(x=test_dates.iloc[0], color='red', linestyle='--', label='Batas Train-Test Split')
        ax_split.set_title(f"Distribusi Historis & Partisi Data - {selected_item}")
        ax_split.set_ylabel("Kuantitas Permintaan")
        ax_split.grid(True, linestyle='--', alpha=0.5)
        ax_split.legend()
        st.pyplot(fig_split)

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("**Statistik Deskriptif Permintaan Mingguan:**")
            st.dataframe(df_model[['Quantity', 'Lag_1', 'Lag_2']].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']], use_container_width=True)
        with col_eda2:
            st.markdown("**Sampel Hasil Transformasi Fitur Time-Series (Lag):**")
            st.dataframe(df_model[['Invoice Date', 'Quantity', 'Lag_1', 'Lag_2', 'MA_Konvensional']].tail(6), use_container_width=True)

    # ------------------------------------------
    # PEMROSESAN INTI SAAT TOMBOL DIKLIK
    # ------------------------------------------
    if not jalankan:
        with tab2:
            st.info("👈 Klik tombol **'🚀 Jalankan Seluruh Alur Model'** di sidebar kiri untuk memulai pelatihan, evaluasi, tuning, dan prediksi stok.")
        with tab3:
            st.info("👈 Klik tombol **'🚀 Jalankan Seluruh Alur Model'** di sidebar kiri untuk melihat hasil Tuning Hiperparameter.")
        with tab4:
            st.info("👈 Klik tombol **'🚀 Jalankan Seluruh Alur Model'** di sidebar kiri untuk melihat hasil Implementasi Model.")
    else:
        with st.spinner("Sedang menjalankan pelatihan model, validasi silang (k=5), tuning hiperparameter, dan peramalan stok..."):
            # Scaling khusus LSTM
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train_sc = scaler_X.fit_transform(X_train)
            X_test_sc = scaler_X.transform(X_test)
            y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

            X_train_lstm = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
            X_test_lstm = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))

            # 0. Metode Konvensional (Moving Average) - Sesuai Bab 3.2.3
            pred_ma = df_model['MA_Konvensional'].iloc[-len(y_test):].values

            # 1. Random Forest Baseline
            rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_base.fit(X_train, y_train)
            pred_rf = np.maximum(rf_base.predict(X_test), 0)

            # 2. SVM Baseline (dengan Regularisasi L2 C=1.0)
            svm_base = SVR(kernel='rbf', C=1.0)
            svm_base.fit(X_train, y_train)
            pred_svm = np.maximum(svm_base.predict(X_test), 0)

            # 3. LSTM Baseline (dengan Dropout)
            lstm_base = build_lstm_model(units=50, dropout_rate=0.2)
            lstm_base.fit(X_train_lstm, y_train_sc, epochs=40, verbose=0)
            pred_lstm = np.maximum(scaler_y.inverse_transform(lstm_base.predict(X_test_lstm, verbose=0)).flatten(), 0)

            # 4. Hybrid Model (LSTM + RF Residual) - Sesuai Bab 3.1.4a
            train_pred_lstm = scaler_y.inverse_transform(lstm_base.predict(X_train_lstm, verbose=0)).flatten()
            residu_train = y_train.values - train_pred_lstm
            rf_residu = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_residu.fit(X_train, residu_train)
            pred_hybrid = np.maximum(pred_lstm + rf_residu.predict(X_test), 0)

            # 5. Voting Regressor (Rata-rata 3 Model) - Sesuai Bab 3.1.4b
            pred_voting = np.maximum((pred_rf + pred_svm + pred_lstm) / 3.0, 0)

            # 6. Weighted Average (Bobot sesuai performa MAE data latih) - Sesuai Bab 3.1.4b
            mae_tr_rf = mean_absolute_error(y_train, rf_base.predict(X_train)) + 1e-5
            mae_tr_svm = mean_absolute_error(y_train, svm_base.predict(X_train)) + 1e-5
            mae_tr_lstm = mean_absolute_error(y_train, train_pred_lstm) + 1e-5
            w_rf, w_svm, w_lstm = 1/mae_tr_rf, 1/mae_tr_svm, 1/mae_tr_lstm
            total_w = w_rf + w_svm + w_lstm
            pred_weighted = np.maximum((w_rf*pred_rf + w_svm*pred_svm + w_lstm*pred_lstm) / total_w, 0)

            # Simpan semua prediksi dalam dictionary
            all_preds = {
                "Metode Konvensional (Moving Average)": pred_ma,
                "Random Forest (RF)": pred_rf,
                "Support Vector Machine (SVM)": pred_svm,
                "Long Short-Term Memory (LSTM)": pred_lstm,
                "Hybrid Model (LSTM + RF)": pred_hybrid,
                "Voting Regressor (Rata-rata 3 Model)": pred_voting,
                "Weighted Average (Bobot Performa)": pred_weighted
            }

            # Hitung tabel evaluasi semua model
            eval_rows = []
            for nama_m, hasil_p in all_preds.items():
                m_mae, m_rmse, m_mape = hitung_metrik(y_test, hasil_p)
                eval_rows.append({
                    "Metode Peramalan": nama_m,
                    "MAE": round(m_mae, 2),
                    "RMSE": round(m_rmse, 2),
                    "MAPE (%)": round(m_mape, 2)
                })
            df_eval_all = pd.DataFrame(eval_rows)

            # Validasi Silang (k-Fold Cross Validation k=5) sesuai Bab 3.2.1
            kf = KFold(n_splits=5, shuffle=False)
            cv_scores_rf, cv_scores_svm = [], []
            for tr_idx, val_idx in kf.split(X):
                X_tr_cv, X_val_cv = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr_cv, y_val_cv = y.iloc[tr_idx], y.iloc[val_idx]
                m_rf_cv = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr_cv, y_tr_cv)
                m_svm_cv = SVR(kernel='rbf', C=1.0).fit(X_tr_cv, y_tr_cv)
                cv_scores_rf.append(mean_absolute_error(y_val_cv, m_rf_cv.predict(X_val_cv)))
                cv_scores_svm.append(mean_absolute_error(y_val_cv, m_svm_cv.predict(X_val_cv)))

        # ------------------------------------------
        # TAB 2: LATIH & EVALUASI MODEL (+ K-FOLD)
        # ------------------------------------------
        with tab2:
            st.markdown('<span class="step-badge">Tahap 2 (Gambar 3.1 & Bab 3.2): Latih Model ➔ Evaluasi Model (MAE, RMSE, MAPE) ➔ K-Fold (k=5) ➔ Komparasi Konvensional</span>', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Hasil Evaluasi Model Fokus: **{model_choice}**")

            pred_fokus = all_preds[model_choice]
            f_mae, f_rmse, f_mape = hitung_metrik(y_test, pred_fokus)
            konv_mae, konv_rmse, konv_mape = hitung_metrik(y_test, pred_ma)

            m1, m2, m3 = st.columns(3)
            m1.metric("📉 MAE (Mean Absolute Error)", f"{f_mae:.2f}", delta=f"{konv_mae - f_mae:.2f} vs Konvensional")
            m2.metric("📊 RMSE (Root Mean Square Error)", f"{f_rmse:.2f}", delta=f"{konv_rmse - f_rmse:.2f} vs Konvensional")
            m3.metric("🎯 MAPE (Mean Absolute Percentage Error)", f"{f_mape:.2f}%", delta=f"{konv_mape - f_mape:.2f}% vs Konvensional")

            st.markdown("#### 📈 Grafik Perbandingan: Data Aktual vs Prediksi AI vs Metode Konvensional")
            fig_eval, ax_eval = plt.subplots(figsize=(12, 4.5))
            ax_eval.plot(test_dates, y_test.values, label='Aktual (Data Gudang Riil)', color='#1E40AF', linewidth=2.5, marker='o')
            ax_eval.plot(test_dates, pred_fokus, label=f'Prediksi AI ({model_choice})', color='#E11D48', linewidth=2.5, linestyle='--', marker='x')
            ax_eval.plot(test_dates, pred_ma, label='Metode Konvensional (Moving Average)', color='#94A3B8', linewidth=1.8, linestyle=':')
            ax_eval.set_ylabel("Kuantitas Permintaan")
            ax_eval.grid(True, linestyle=':', alpha=0.7)
            ax_eval.legend()
            st.pyplot(fig_eval)

            st.divider()
            col_t1, col_t2 = st.columns([3, 2])
            with col_t1:
                st.markdown("#### 🏆 Tabel Komparasi Seluruh Metode (AI vs Konvensional)")
                st.dataframe(df_eval_all, use_container_width=True)
            with col_t2:
                st.markdown("#### 🔄 Hasil Validasi Silang (5-Fold Cross Validation)")
                st.write(f"- **Rata-rata MAE 5-Fold (Random Forest):** `{np.mean(cv_scores_rf):.2f}` (± {np.std(cv_scores_rf):.2f})")
                st.write(f"- **Rata-rata MAE 5-Fold (SVM L2):** `{np.mean(cv_scores_svm):.2f}` (± {np.std(cv_scores_svm):.2f})")
                st.caption("Sesuai Bab 3.2.1: K-Fold Cross Validation (k=5) memastikan model mampu menggeneralisasi pola pada berbagai partisi waktu dan menghindari overfitting.")

        # ------------------------------------------
        # TAB 3: TUNING HIPERPARAMETER (GAMBAR 3.1)
        # ------------------------------------------
        with tab3:
            st.markdown('<span class="step-badge">Tahap 3 (Gambar 3.1): Evaluasi Model ➔ Tuning Hiperparameter</span>', unsafe_allow_html=True)
            st.markdown("### 🔧 Optimasi & Tuning Hiperparameter Model")
            st.write("Sesuai alur pada **Gambar 3.1** dan **Subbab 3.2.1**, tahap ini menguji perubahan hiperparameter (jumlah pohon *Random Forest*, parameter regularisasi L2 pada *SVM*, dan *Dropout* pada *LSTM*) untuk meminimalkan *error* prediksi.")

            # Jalankan model dengan parameter hasil tuning dari sidebar
            rf_tuned = RandomForestRegressor(n_estimators=rf_trees_tune, max_depth=8, min_samples_split=3, random_state=42).fit(X_train, y_train)
            pred_rf_tuned = np.maximum(rf_tuned.predict(X_test), 0)

            svm_tuned = SVR(kernel='rbf', C=svm_c_tune, gamma='scale').fit(X_train, y_train)
            pred_svm_tuned = np.maximum(svm_tuned.predict(X_test), 0)

            lstm_tuned = build_lstm_model(units=64, dropout_rate=lstm_dropout)
            lstm_tuned.fit(X_train_lstm, y_train_sc, epochs=50, verbose=0)
            pred_lstm_tuned = np.maximum(scaler_y.inverse_transform(lstm_tuned.predict(X_test_lstm, verbose=0)).flatten(), 0)

            # Hybrid Tuned
            tr_pred_lstm_t = scaler_y.inverse_transform(lstm_tuned.predict(X_train_lstm, verbose=0)).flatten()
            rf_res_tuned = RandomForestRegressor(n_estimators=rf_trees_tune, max_depth=8, random_state=42).fit(X_train, y_train.values - tr_pred_lstm_t)
            pred_hybrid_tuned = np.maximum(pred_lstm_tuned + rf_res_tuned.predict(X_test), 0)

            tuning_comparison = []
            for label_m, p_before, p_after, param_info in [
                ("Random Forest (RF)", pred_rf, pred_rf_tuned, f"n_estimators={rf_trees_tune}, max_depth=8"),
                ("Support Vector Machine (SVM)", pred_svm, pred_svm_tuned, f"Kernel=RBF, L2 Regularization C={svm_c_tune}"),
                ("LSTM (Deep Learning)", pred_lstm, pred_lstm_tuned, f"Units=64, Dropout={lstm_dropout}, Epochs=50"),
                ("Hybrid Model (LSTM + RF)", pred_hybrid, pred_hybrid_tuned, f"LSTM Dropout={lstm_dropout} + RF Trees={rf_trees_tune}")
            ]:
                mae_b, rmse_b, _ = hitung_metrik(y_test, p_before)
                mae_a, rmse_a, _ = hitung_metrik(y_test, p_after)
                tuning_comparison.append({
                    "Algoritma": label_m,
                    "Konfigurasi Parameter Tuning": param_info,
                    "MAE Sebelum Tuning": round(mae_b, 2),
                    "MAE Sesudah Tuning": round(mae_a, 2),
                    "RMSE Sebelum Tuning": round(rmse_b, 2),
                    "RMSE Sesudah Tuning": round(rmse_a, 2)
                })

            st.dataframe(pd.DataFrame(tuning_comparison), use_container_width=True)

        # ------------------------------------------
        # TAB 4: IMPLEMENTASI MODEL (GAMBAR 3.1)
        # ------------------------------------------
        with tab4:
            st.markdown('<span class="step-badge">Tahap 4 (Gambar 3.1): Implementasi Model ➔ Rekomendasi Pengelolaan Stok Gudang ➔ Selesai</span>', unsafe_allow_html=True)
            st.markdown(f"### 📦 Peramalan Kebutuhan Stok {horizon_weeks} Minggu ke Depan: **{selected_item}**")

            # Prediksi iteratif N minggu ke depan menggunakan model fokus
            future_preds = []
            last_lag1 = float(df_model['Quantity'].iloc[-1])
            last_lag2 = float(df_model['Quantity'].iloc[-2])
            last_date = df_model['Invoice Date'].iloc[-1]
            future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(horizon_weeks)]

            for _ in range(horizon_weeks):
                x_next = pd.DataFrame([[last_lag1, last_lag2]], columns=['Lag_1', 'Lag_2'])
                x_next_sc = scaler_X.transform(x_next).reshape((1, 1, 2))

                p_rf_n = float(rf_tuned.predict(x_next)[0])
                p_svm_n = float(svm_tuned.predict(x_next)[0])
                p_lstm_n = float(scaler_y.inverse_transform(lstm_tuned.predict(x_next_sc, verbose=0))[0][0])
                p_res_n = float(rf_res_tuned.predict(x_next)[0])

                if model_choice == "Random Forest (RF)":
                    val_next = p_rf_n
                elif model_choice == "Support Vector Machine (SVM)":
                    val_next = p_svm_n
                elif model_choice == "Long Short-Term Memory (LSTM)":
                    val_next = p_lstm_n
                elif model_choice == "Hybrid Model (LSTM + RF)":
                    val_next = p_lstm_n + p_res_n
                elif model_choice == "Voting Regressor (Rata-rata 3 Model)":
                    val_next = (p_rf_n + p_svm_n + p_lstm_n) / 3.0
                else:
                    val_next = (w_rf*p_rf_n + w_svm*p_svm_n + w_lstm*p_lstm_n) / total_w

                val_next = max(0.0, round(val_next, 0))
                future_preds.append(val_next)
                last_lag2 = last_lag1
                last_lag1 = val_next

            # Hitung Safety Stock & Rekomendasi Pengadaan Gudang
            std_hist = float(df_model['Quantity'].tail(12).std())
            safety_stock = round(1.65 * std_hist, 0) if not np.isnan(std_hist) else 0.0

            df_future = pd.DataFrame({
                "Minggu Ke-": [f"Minggu +{i+1}" for i in range(horizon_weeks)],
                "Tanggal Periode": [d.strftime('%Y-%m-%d') for d in future_dates],
                "Prediksi Permintaan (Unit)": future_preds,
                "Safety Stock (Buffer)": [safety_stock] * horizon_weeks,
                "Rekomendasi Stok Optimal (Unit)": [round(p + safety_stock, 0) for p in future_preds]
            })

            c_f1, c_f2, c_f3 = st.columns(3)
            c_f1.metric("Total Prediksi Permintaan", f"{int(sum(future_preds)):,} Unit", f"{horizon_weeks} Minggu ke Depan")
            c_f2.metric("Rekomendasi Safety Stock", f"{int(safety_stock):,} Unit", "Cegah Risiko Stockout")
            c_f3.metric("Batas Maksimal Pengadaan", f"{int(sum(future_preds) + safety_stock):,} Unit", "Cegah Risiko Overstock")

            # Grafik Historis + Forecast Masa Depan
            fig_fut, ax_fut = plt.subplots(figsize=(12, 4.5))
            hist_tail = df_model.tail(24)
            ax_fut.plot(hist_tail['Invoice Date'], hist_tail['Quantity'], label='Data Historis Terakhir (24 Minggu)', color='#1E40AF', marker='o')
            ax_fut.plot(future_dates, future_preds, label=f'Forecast Masa Depan ({model_choice})', color='#16A34A', linewidth=2.5, linestyle='--', marker='s')
            ax_fut.fill_between(future_dates, future_preds, [p + safety_stock for p in future_preds], color='#86EFAC', alpha=0.35, label='Area Safety Stock Optimal')
            ax_fut.set_title(f"Proyeksi Kebutuhan Stok Gudang ke Depan - {selected_item}")
            ax_fut.set_ylabel("Kuantitas (Unit)")
            ax_fut.grid(True, linestyle='--', alpha=0.5)
            ax_fut.legend()
            st.pyplot(fig_fut)

            st.markdown("#### 📋 Tabel Rencana Pengadaan Stok Gudang PT Samudra Utama Narapati")
            st.dataframe(df_future, use_container_width=True)

else:
    st.markdown("""
        <div style='text-align: center; padding: 45px; background-color: #F8FAFC; border-radius: 12px; border: 1px dashed #94A3B8;'>
            <h3 style='color: #1E3A8A;'>📦 Sistem Siap Digunakan</h3>
            <p style='color: #475569;'>Silakan unggah file <b>Data Penjualan Historis (.xls / .xlsx)</b> pada panel sebelah kiri untuk menjalankan seluruh tahapan sesuai Gambar 3.1 Proposal Tugas Akhir.</p>
        </div>
    """, unsafe_allow_html=True)
