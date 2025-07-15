#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 FY102HF LSTM 并导出：
    model.h5           – 网络
    scaler.gz          – MinMaxScaler
    meta.json          – 特征列顺序、window_size、steps_ahead
    LSTM_loss.png      – 损失曲线
    LSTM_predictions.xlsx – 预测结果
"""

import argparse, os, re, itertools, json, joblib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# 0 — 参数
BASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=str(BASE_DIR / "FY102HF.csv"), help="CSV 文件路径")
args = parser.parse_args()

# 1 — 数据与特征工程（与你原逻辑一致，略）

series = pd.read_csv(args.csv, encoding="gbk", parse_dates=["DATEPRD"], index_col="DATEPRD")

series.insert(0, "days", pd.Series(range(len(series), 0, -1), index=series.index))
series["days(t)"] = series["days"].shift(-1)
for col in ["CHOKE_SIZE", "ON_STREAM_HRS", "Pmax", "Pmin",
            "Tmax", "Tmin", "BORE_water_VOL", "BORE_GAS_VOL", "BORE_OIL_VOL"]:
    series[f"{col}(t)"] = series[col].shift(-1)

series["interaction_effect_onNext_oilRate"] = (
    series["CHOKE_SIZE(t)"] * series["ON_STREAM_HRS(t)"] * series["days(t)"]
)
series.dropna(inplace=True)

feature_cols = [
    "CHOKE_SIZE(t)", "ON_STREAM_HRS(t)", "Pmin(t)", "Pmax(t)",
    "Tmin(t)", "Tmax(t)", "BORE_water_VOL(t)", "BORE_GAS_VOL(t)"
]
target_col   = "BORE_OIL_VOL"
matrix       = series[feature_cols + [target_col]]

# 2 — 转监督函数
def series_to_supervised(df, cols, n_in, n_out, dropnan=True):
    agg, names = [], []
    for i in range(n_in, 0, -1):
        agg.append(df.shift(i)); names += [f"{c}(t-{i})" for c in cols]
    for i in range(n_out):
        agg.append(df.shift(-i)); names += [f"{c}(t+{i})" for c in cols]
    out = pd.concat(agg, axis=1); out.columns = names
    if dropnan: out.dropna(inplace=True)
    return out

# 3 — 网格搜索（保持原搜索空间）
steps_ahead = 1
grid = list(itertools.product([500], [1, 2], [20, 40, 80, 160], [2, 4], [8]))

os.environ["PYTHONHASHSEED"] = "0"; np.random.seed(42); tf.random.set_seed(42)
pattern = re.compile(r"(t-)|^BORE_OIL_VOL.*")

best_loss = np.inf
for epochs, layers, units, batch, window in grid:
    sup = series_to_supervised(matrix, matrix.columns, n_in=window, n_out=steps_ahead)
    sup = sup[[c for c in sup.columns if re.search(pattern, c)]]

    n_feat = int((sup.shape[1] - steps_ahead) / window)
    data   = sup.values
    split  = int(len(data) * 0.8)
    tr, te = data[:split], data[split:]

    scaler = MinMaxScaler(); tr_s = scaler.fit_transform(tr); te_s = scaler.transform(te)
    trX = tr_s[:, :-steps_ahead].reshape(-1, window, n_feat); trY = tr_s[:, -steps_ahead:]
    teX = te_s[:, :-steps_ahead].reshape(-1, window, n_feat); teY = te_s[:, -steps_ahead:]

    model = Sequential()
    if layers == 1:
        model.add(LSTM(units, activation="tanh", input_shape=(window, n_feat)))
    else:
        for _ in range(layers - 1):
            model.add(LSTM(units, activation="tanh", return_sequences=True,
                           input_shape=(window, n_feat)))
        model.add(LSTM(units))
    model.add(Dense(steps_ahead))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(1e-4))
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    hist = model.fit(trX, trY, epochs=epochs, batch_size=batch,
                     validation_data=(teX, teY), callbacks=[es],
                     verbose=0, shuffle=False)

    loss = model.evaluate(teX, teY, verbose=0)
    if loss < best_loss:
        best_loss, best_model, best_hist = loss, model, hist
        best_scaler, best_window, best_nfeat = scaler, window, n_feat
        best_params = dict(epochs=epochs, layers=layers,
                           units=units, batch=batch, window=window)

print("Best params:", best_params)

# 4 — 保存 loss 曲线
plt.plot(best_hist.history["loss"], label="train")
plt.plot(best_hist.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
plt.savefig("LSTM_loss.png", dpi=300, bbox_inches="tight"); plt.close()

# 5 — 预测并导出 Excel
sup = series_to_supervised(matrix, matrix.columns, n_in=best_window, n_out=steps_ahead)
sup = sup[[c for c in sup.columns if re.search(pattern, c)]]
data = sup.values; split = int(len(data) * 0.8)
tr, te = data[:split], data[split:]
tr_s = best_scaler.transform(tr); te_s = best_scaler.transform(te)
trX = tr_s[:, :-steps_ahead].reshape(-1, best_window, best_nfeat); trY = tr_s[:, -steps_ahead:]
teX = te_s[:, :-steps_ahead].reshape(-1, best_window, best_nfeat); teY = te_s[:, -steps_ahead:]

pred_tr = best_model.predict(trX); pred_te = best_model.predict(teX)
ymin, ymax = best_scaler.data_min_[-1], best_scaler.data_max_[-1]
inv = lambda y: y*(ymax - ymin) + ymin
trY, teY   = inv(trY), inv(teY)
pred_tr, pred_te = inv(pred_tr), inv(pred_te)

with pd.ExcelWriter("LSTM_predictions.xlsx") as writer:
    pd.DataFrame({"Actual": trY.flatten(), "Pred": pred_tr.flatten()}
                 ).to_excel(writer, sheet_name="Train", index=False)
    pd.DataFrame({"Actual": teY.flatten(), "Pred": pred_te.flatten()}
                 ).to_excel(writer, sheet_name="Test",  index=False)

# 6 — 保存模型 / scaler / meta.json
best_model.save("model.h5")
joblib.dump(best_scaler, "scaler.gz")

meta = {
    "feature_cols": feature_cols,
    "window_size": best_window,
    "steps_ahead": steps_ahead,
    "best_params": best_params
}
json.dump(meta, open("meta.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("Exported: model.h5  scaler.gz  meta.json  LSTM_loss.png  LSTM_predictions.xlsx")
