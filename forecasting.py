import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# ==========================================
# KONFIGURASI FILE EXCEL
# ==========================================
FILE_PATH = "Data_Penjualan_SUN_upload.xlsx"  # Ubah sesuai nama file Excel Anda
SHEET_NAME = "Rekap"

TARGET_4_SKUS = [
    "NARAXONE KEMASAN 1 LITER",
    "NARATOP KEMASAN 1 LITER",
    "NARALLY KEMASAN 5 GRAM",
    "NARAPON KEMASAN 500 ML (WHITE)",
]

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 1. LOAD, CLEAN (SESUAI TABEL 2) & AGREGASI
# ==========================================
def load_and_clean_data(file_path, sheet_name="Rekap"):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    df_raw["Invoice Date"] = pd.to_datetime(df_raw["Invoice Date"], errors="coerce")
    df_raw["Quantity"] = pd.to_numeric(df_raw["Quantity"], errors="coerce")

    # Pemisahan transaksi retur/koreksi (Quantity <= 0)
    df_pos = df_raw[df_raw["Quantity"] > 0].copy()

    # Eliminasi duplikasi identik pada 4 atribut utama (jika Invoice No. tersedia)
    dup_cols = ["Invoice No.", "Invoice Date", "Item Description", "Quantity"]
    if all(c in df_pos.columns for c in dup_cols):
        df_clean = df_pos.drop_duplicates(subset=dup_cols).copy()
    else:
        df_clean = df_pos.drop_duplicates().copy()

    full_weeks = pd.date_range(
        start=df_clean["Invoice Date"].min(),
        end=df_clean["Invoice Date"].max(),
        freq="W",
    )
    weekly_pivot = (
        df_clean.groupby([pd.Grouper(key="Invoice Date", freq="W"), "Item Description"])[
            "Quantity"
        ]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(full_weeks, fill_value=0.0)
    )
    print(f"Baris bersih: {len(df_clean)} | Total SKU: {weekly_pivot.shape[1]} | Total Minggu: {len(weekly_pivot)}")
    return weekly_pivot


# ==========================================
# 2. KLASIFIKASI ADI-CV² (109 MINGGU LATIH)
# ==========================================
def classify_skus(weekly_pivot):
    train_df = weekly_pivot.iloc[2:111]  # 109 minggu latih
    records = []
    for sku in train_df.columns:
        y_tr = train_df[sku].values
        nz_tr = y_tr[y_tr > 0]
        n_nz = len(nz_tr)
        denom = np.mean(np.abs(np.diff(y_tr)))

        if n_nz == 0 or denom <= 0:
            pat = "Inactive"
            adi, cv2 = np.nan, np.nan
        else:
            adi = len(y_tr) / n_nz
            cv2 = (np.std(nz_tr, ddof=1) / np.mean(nz_tr)) ** 2 if n_nz > 1 else 0.0
            if adi <= 1.32 and cv2 <= 0.49:
                pat = "Smooth"
            elif adi <= 1.32 and cv2 > 0.49:
                pat = "Erratic"
            elif adi > 1.32 and cv2 <= 0.49:
                pat = "Intermittent"
            else:
                pat = "Lumpy"

        records.append(
            {"SKU": sku, "Pattern": pat, "ADI": adi, "CV2": cv2, "MASE_Denom": denom}
        )
    return pd.DataFrame(records)


# ==========================================
# 3. FITUR & BASELINE STATISTIK CAUSAL (h=1)
# ==========================================
def make_features(series):
    df_f = pd.DataFrame({"y": series.values})
    df_f["Lag_1"] = df_f["y"].shift(1)
    df_f["Lag_2"] = df_f["y"].shift(2)
    df_f["Lag_4"] = df_f["y"].shift(4)
    df_f["Roll_Mean_4"] = df_f["y"].shift(1).rolling(4, min_periods=1).mean()
    df_f["Roll_Std_4"] = df_f["y"].shift(1).rolling(4, min_periods=1).std().fillna(0.0)
    df_eff = df_f.iloc[2:].fillna(0.0)
    return df_eff.drop(columns=["y"]).values, df_eff["y"].values


def causal_ets(y_full, alpha=0.35):
    preds = np.zeros(len(y_full))
    level = y_full[0]
    preds[0] = level
    for t in range(1, len(y_full)):
        preds[t] = level
        level = alpha * y_full[t] + (1.0 - alpha) * level
    return np.maximum(0.0, preds[2:])  # sejajar dengan df_eff (137 minggu)


def causal_croston_sba(y_full, alpha=0.25):
    preds = np.zeros(len(y_full))
    z, p, q = 0.0, 1.0, 1
    initialized = False
    for t in range(len(y_full)):
        preds[t] = (1.0 - alpha / 2.0) * (z / p) if initialized else 0.0
        if y_full[t] > 0:
            if not initialized:
                z = float(y_full[t])
                p = float(q)
                initialized = True
            else:
                z = alpha * y_full[t] + (1.0 - alpha) * z
                p = alpha * q + (1.0 - alpha) * p
            q = 1
        else:
            q += 1
    return np.maximum(0.0, preds[2:])  # sejajar dengan df_eff (137 minggu)


def fit_predict_lstm(X_tr_s, y_tr_s, X_eval_s, sc_y, epochs=30):
    model = Sequential(
        [
            LSTM(32, activation="relu", input_shape=(X_tr_s.shape[1], 1)),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_tr_s.reshape(-1, X_tr_s.shape[1], 1),
        y_tr_s,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        shuffle=False,
    )
    raw_pred = model(X_eval_s.reshape(-1, X_eval_s.shape[1], 1), training=False).numpy()
    return np.maximum(0.0, sc_y.inverse_transform(raw_pred).ravel())


# ==========================================
# 4. EVALUASI 10 MODEL PADA 136 SKU AKTIF
# ==========================================
weekly_pivot = load_and_clean_data(FILE_PATH, SHEET_NAME)
df_cls = classify_skus(weekly_pivot)
active_df = df_cls[df_cls["Pattern"] != "Inactive"].reset_index(drop=True)
print(f"SKU Aktif dievaluasi: {len(active_df)}")
print(active_df["Pattern"].value_counts(), "\n")

MODEL_NAMES = [
    "Naive", "MA-4", "ETS", "Croston-SBA",
    "RF", "SVR", "LSTM", "Hybrid", "Voting", "Weighted"
]

val_rows = []
test_rows = []
sim_rows = []

for idx, row in active_df.iterrows():
    sku = row["SKU"]
    pat = row["Pattern"]
    denom = row["MASE_Denom"]

    series = weekly_pivot[sku]
    y_full = series.values
    X, y = make_features(series)
    X_train, y_train = X[:109], y[:109]
    X_test, y_test = X[109:], y[109:]

    # Baseline statistik causal (137 minggu -> split 109 latih, 28 uji)
    p_naive_all = X[:, 0]
    p_ma4_all = X[:, 3]
    p_ets_all = causal_ets(y_full, alpha=0.35)
    p_sba_all = causal_croston_sba(y_full, alpha=0.25)

    # 5-Fold Rolling-Origin CV pada 109 minggu latih
    tscv = TimeSeriesSplit(n_splits=5)
    oof_indices, oof_y, oof_X = [], [], []
    oof_rf, oof_svr, oof_lstm, oof_hyb = [], [], [], []
    prev_X_list, prev_res_list = [], []

    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_t, y_t = X_train[tr_idx], y_train[tr_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]

        sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
        X_tr_s = sc_X.fit_transform(X_t)
        X_val_s = sc_X.transform(X_v)
        y_tr_s = sc_y.fit_transform(y_t.reshape(-1, 1)).ravel()
        sw = np.exp(np.linspace(-1.8, 0, len(y_t)))

        rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
        rf.fit(X_t, y_t, sample_weight=sw)
        p_rf_v = np.maximum(0.0, rf.predict(X_v))

        svr = SVR(kernel="rbf", C=1.5, epsilon=0.05, gamma="scale")
        svr.fit(X_tr_s, y_tr_s, sample_weight=sw)
        p_svr_v = np.maximum(0.0, sc_y.inverse_transform(svr.predict(X_val_s).reshape(-1, 1)).ravel())

        p_lstm_v = fit_predict_lstm(X_tr_s, y_tr_s, X_val_s, sc_y, epochs=30)

        # Hybrid OOF (bebas leakage: latih residual RF hanya dari fold validasi sebelumnya)
        if fold_i == 0:
            p_hyb_v = p_lstm_v.copy()
        else:
            X_res_tr = np.vstack(prev_X_list)
            y_res_tr = np.concatenate(prev_res_list)
            rf_res_cv = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=42)
            rf_res_cv.fit(X_res_tr, y_res_tr)
            p_hyb_v = np.maximum(0.0, p_lstm_v + rf_res_cv.predict(X_v))

        prev_X_list.append(X_v)
        prev_res_list.append(y_v - p_lstm_v)

        oof_indices.extend(val_idx)
        oof_y.extend(y_v)
        oof_X.append(X_v)
        oof_rf.extend(p_rf_v)
        oof_svr.extend(p_svr_v)
        oof_lstm.extend(p_lstm_v)
        oof_hyb.extend(p_hyb_v)

    oof_idx = np.array(oof_indices)
    oof_y = np.array(oof_y)
    oof_X_arr = np.vstack(oof_X)
    oof_rf, oof_svr, oof_lstm, oof_hyb = map(np.array, [oof_rf, oof_svr, oof_lstm, oof_hyb])

    # Bobot Weighted Ensemble dari MAE validasi temporal (Persamaan 2)
    mae_v_rf = np.mean(np.abs(oof_y - oof_rf)) + 1e-6
    mae_v_svr = np.mean(np.abs(oof_y - oof_svr)) + 1e-6
    mae_v_lstm = np.mean(np.abs(oof_y - oof_lstm)) + 1e-6
    inv_sum = (1 / mae_v_rf) + (1 / mae_v_svr) + (1 / mae_v_lstm)
    w_rf, w_svr, w_lstm = (1 / mae_v_rf) / inv_sum, (1 / mae_v_svr) / inv_sum, (1 / mae_v_lstm) / inv_sum

    oof_vot = (oof_rf + oof_svr + oof_lstm) / 3.0
    oof_wgt = w_rf * oof_rf + w_svr * oof_svr + w_lstm * oof_lstm

    val_preds = {
        "Naive": p_naive_all[oof_idx],
        "MA-4": p_ma4_all[oof_idx],
        "ETS": p_ets_all[oof_idx],
        "Croston-SBA": p_sba_all[oof_idx],
        "RF": oof_rf,
        "SVR": oof_svr,
        "LSTM": oof_lstm,
        "Hybrid": oof_hyb,
        "Voting": oof_vot,
        "Weighted": oof_wgt,
    }
    val_entry = {"SKU": sku, "Pattern": pat}
    for m in MODEL_NAMES:
        val_entry[m] = np.mean(np.abs(oof_y - val_preds[m])) / denom
    val_rows.append(val_entry)

    # Latih penuh pada 109 minggu & uji pada 28 minggu holdout
    sc_X_f, sc_y_f = MinMaxScaler(), MinMaxScaler()
    X_tr_fs = sc_X_f.fit_transform(X_train)
    X_te_fs = sc_X_f.transform(X_test)
    y_tr_fs = sc_y_f.fit_transform(y_train.reshape(-1, 1)).ravel()
    sw_f = np.exp(np.linspace(-1.8, 0, len(y_train)))

    rf_f = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42).fit(X_train, y_train, sample_weight=sw_f)
    p_rf_te = np.maximum(0.0, rf_f.predict(X_test))

    svr_f = SVR(kernel="rbf", C=1.5, epsilon=0.05, gamma="scale").fit(X_tr_fs, y_tr_fs, sample_weight=sw_f)
    p_svr_te = np.maximum(0.0, sc_y_f.inverse_transform(svr_f.predict(X_te_fs).reshape(-1, 1)).ravel())

    p_lstm_te = fit_predict_lstm(X_tr_fs, y_tr_fs, X_te_fs, sc_y_f, epochs=45)

    rf_res_f = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=42).fit(oof_X_arr, oof_y - oof_lstm)
    p_hyb_te = np.maximum(0.0, p_lstm_te + rf_res_f.predict(X_test))

    p_vot_te = (p_rf_te + p_svr_te + p_lstm_te) / 3.0
    p_wgt_te = w_rf * p_rf_te + w_svr * p_svr_te + w_lstm * p_lstm_te

    test_preds = {
        "Naive": p_naive_all[109:],
        "MA-4": p_ma4_all[109:],
        "ETS": p_ets_all[109:],
        "Croston-SBA": p_sba_all[109:],
        "RF": p_rf_te,
        "SVR": p_svr_te,
        "LSTM": p_lstm_te,
        "Hybrid": p_hyb_te,
        "Voting": p_vot_te,
        "Weighted": p_wgt_te,
    }
    test_entry = {"SKU": sku, "Pattern": pat}
    for m in MODEL_NAMES:
        test_entry[m] = np.mean(np.abs(y_test - test_preds[m])) / denom
    test_rows.append(test_entry)

    # Verifikasi Simulasi Persediaan 4 SKU (Poin 1.1)
    if sku in TARGET_4_SKUS:
        sigma_e_val = np.std(oof_y - oof_wgt, ddof=1)
        ss = 1.65 * sigma_e_val * np.sqrt(2.0)
        d_hat_mean = np.mean(p_wgt_te)
        rop_mean = (d_hat_mean * 2.0) + ss

        # Versi A: Perbandingan mingguan dinamis (y_t > y_hat_t + SS) — bebas look-ahead
        so_dyn = int(np.sum(y_test > (p_wgt_te + ss)))
        csl_dyn = round(((len(y_test) - so_dyn) / len(y_test)) * 100, 2)

        # Versi B: Sesuai skrip Streamlit awal (y_t > mean(y_hat) + SS)
        so_const = int(np.sum(y_test > (d_hat_mean + ss)))
        csl_const = round(((len(y_test) - so_const) / len(y_test)) * 100, 2)

        sim_rows.append(
            {
                "SKU": sku,
                "Pattern": pat,
                "Sigma_e_Val": round(sigma_e_val, 2),
                "Safety_Stock": round(ss, 2),
                "ROP": round(rop_mean, 2),
                "Stockout_Mingguan_Dinamis": f"{so_dyn}/28",
                "CSL_Dinamis_%": csl_dyn,
                "Stockout_Versi_Mean": f"{so_const}/28",
                "CSL_Versi_Mean_%": csl_const,
            }
        )

    tf.keras.backend.clear_session()
    if (idx + 1) % 20 == 0 or (idx + 1) == len(active_df):
        print(f"Progres: {idx + 1}/{len(active_df)} SKU selesai...")

# ==========================================
# 5. REKAP HASIL UNTUK REVISI JURNAL
# ==========================================
df_val = pd.DataFrame(val_rows)
df_test = pd.DataFrame(test_rows)
df_sim = pd.DataFrame(sim_rows)

order_pat = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
val_median_tbl = df_val.groupby("Pattern")[MODEL_NAMES].median().reindex(order_pat).round(3)
test_median_tbl = df_test.groupby("Pattern")[MODEL_NAMES].median().reindex(order_pat).round(3)

# Model terpilih MURNI dari Data Validasi (Poin 1.4)
selected_by_val = val_median_tbl.idxmin(axis=1)

# Evaluasi Gabungan 136 SKU pada Data Uji Holdout (Pattern-Aware vs Model Tunggal)
df_test["Pattern_Aware_ValSelected"] = df_test.apply(
    lambda r: r[selected_by_val[r["Pattern"]]], axis=1
)
summary_cols = MODEL_NAMES + ["Pattern_Aware_ValSelected"]
combined_eval = pd.DataFrame(
    {
        "Median_MASE_136SKU_Test": df_test[summary_cols].median().round(3),
        "Mean_MASE_136SKU_Test": df_test[summary_cols].mean().round(3),
    }
).sort_values("Median_MASE_136SKU_Test")

print("\n=== [POIN 1.4a] TABEL MEDIAN MASE DATA VALIDASI (5-FOLD ROLLING-ORIGIN CV) ===")
print(val_median_tbl)
print("\nModel Terpilih Berdasarkan Data Validasi:")
print(selected_by_val)

print("\n=== [POIN 1.4b] TABEL MEDIAN MASE DATA UJI (28 MINGGU HOLDOUT) ===")
print(test_median_tbl)

print("\n=== [POIN 1.4c] EVALUASI GABUNGAN 136 SKU PADA DATA UJI (PATTERN-AWARE VS TUNGGAL) ===")
print(combined_eval)

print("\n=== [POIN 1.1] VERIFIKASI SIMULASI PERSEDIAAN 4 SKU (CSL / MINGGU TANPA STOCKOUT) ===")
print(df_sim.to_string(index=False))

# Simpan ke Excel agar mudah dicek dan dikirim balik
with pd.ExcelWriter("Hasil_Revisi_Sistemasi_7178.xlsx") as writer:
    val_median_tbl.to_excel(writer, sheet_name="Val_Median_MASE")
    test_median_tbl.to_excel(writer, sheet_name="Test_Median_MASE")
    combined_eval.to_excel(writer, sheet_name="Combined_136SKU")
    df_sim.to_excel(writer, sheet_name="Simulasi_4SKU", index=False)
print("\nFile 'Hasil_Revisi_Sistemasi_7178.xlsx' berhasil disimpan!")
