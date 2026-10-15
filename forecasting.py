import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import Jaringan Saraf Tiruan diletakkan langsung di atas
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Konfigurasi Halaman
st.set_page_config(page_title="Demand Forecasting App", layout="wide")
st.title("Aplikasi Demand Forecasting (Sesuai Proposal TA)")
st.subheader("Data: PT Samudra Utama Narapati")

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    df_grouped = df.groupby(['Invoice Date', 'Item Description'])['Quantity'].sum().reset_index()
    return df_grouped

uploaded_file = st.sidebar.file_uploader("Upload Data_Penjualan_Bersih.xlsx", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_all = load_and_preprocess_data(uploaded_file)
    st.success("Data berhasil dimuat!")

    st.sidebar.header("Pilih Produk")
    item_list = df_all['Item Description'].unique()
    selected_item = st.sidebar.selectbox("Pilih Item untuk Diprediksi:", item_list)

    df_item = df_all[df_all['Item Description'] == selected_item].copy()
    df_item.set_index('Invoice Date', inplace=True)
    df_weekly = df_item['Quantity'].resample('W').sum().fillna(0).reset_index()
    
    st.write(f"### Riwayat Penjualan Mingguan: **{selected_item}**")
    st.line_chart(df_weekly.set_index('Invoice Date'))

    df_weekly['Lag_1'] = df_weekly['Quantity'].shift(1)
    df_weekly['Lag_2'] = df_weekly['Quantity'].shift(2)
    df_weekly.dropna(inplace=True)

    X = df_weekly[['Lag_1', 'Lag_2']]
    y = df_weekly['Quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    st.sidebar.header("Pengaturan Model")
    model_choice = st.sidebar.radio("Pilih Algoritma (Sesuai Bab III):", [
        "Random Forest", 
        "Support Vector Machine (SVM)",
        "Long Short-Term Memory (LSTM)",
        "Hybrid Model (LSTM + Random Forest)",
        "Voting Regressor (Rata-rata 3 Model)"
    ])

    if st.sidebar.button("Jalankan Prediksi"):
        with st.spinner(f'Sedang melatih model {model_choice}... Harap tunggu.'):
            
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            def build_lstm():
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(1, 2)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                return model

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            svm_model = SVR(kernel='rbf')
            y_pred = None

            if model_choice == "Random Forest":
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)

            elif model_choice == "Support Vector Machine (SVM)":
                svm_model.fit(X_train, y_train)
                y_pred = svm_model.predict(X_test)

            elif model_choice == "Long Short-Term Memory (LSTM)":
                model = build_lstm()
                model.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                y_pred_scaled = model.predict(X_test_lstm)
                y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            elif model_choice == "Hybrid Model (LSTM + Random Forest)":
                model = build_lstm()
                model.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                
                train_pred_scaled = model.predict(X_train_lstm)
                train_pred = scaler_y.inverse_transform(train_pred_scaled).flatten()
                residu_train = y_train.values - train_pred
                
                rf_model.fit(X_train, residu_train)
                
                test_pred_scaled = model.predict(X_test_lstm)
                test_pred_lstm = scaler_y.inverse_transform(test_pred_scaled).flatten()
                test_pred_residu = rf_model.predict(X_test)
                y_pred = test_pred_lstm + test_pred_residu

            elif model_choice == "Voting Regressor (Rata-rata 3 Model)":
                rf_model.fit(X_train, y_train)
                pred_rf = rf_model.predict(X_test)
                
                svm_model.fit(X_train, y_train)
                pred_svm = svm_model.predict(X_test)
                
                model = build_lstm()
                model.fit(X_train_lstm, y_train_scaled, epochs=50, verbose=0)
                pred_lstm_scaled = model.predict(X_test_lstm)
                pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()
                
                y_pred = (pred_rf + pred_svm + pred_lstm) / 3

            # Hilangkan kemungkinan nilai prediksi negatif
            y_pred = np.maximum(y_pred, 0)
            
            # Evaluasi metrik
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if sum(mask) > 0 else 0

            st.write(f"### Hasil Evaluasi Model: {model_choice}")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", round(mae, 2))
            col2.metric("RMSE", round(rmse, 2))
            col3.metric("MAPE", f"{round(mape, 2)}%")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_weekly['Invoice Date'].iloc[-len(y_test):], y_test.values, label='Aktual', marker='o')
            ax.plot(df_weekly['Invoice Date'].iloc[-len(y_test):], y_pred, label='Prediksi AI', color='red', linestyle='dashed', marker='x')
            ax.legend()
            st.pyplot(fig)
