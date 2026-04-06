import os
os.system('pip install tensorflow-cpu')
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import plotly.graph_objects as go
import h5py
import json
import keras
from keras.models import model_from_json

def load_legacy_model(model_path):
    with h5py.File(model_path, "r") as f:
        model_config = f.attrs.get("model_config")
        
        if hasattr(model_config, 'decode'):
            config_dict = json.loads(model_config.decode("utf-8"))
        else:
            config_dict = json.loads(model_config)

        # FIX KRUSIAL: Memperbaiki InputLayer dan parameter Keras 3
        if "config" in config_dict and "layers" in config_dict["config"]:
            layers = config_dict["config"]["layers"]
            for layer in layers:
                if "config" in layer:
                    # 1. Perbaikan InputLayer (Penyebab error 'shape' argument)
                    if layer["class_name"] == "InputLayer":
                        # Keras 3 butuh batch_shape. Kita set (None, 120, 1) 
                        # None = batch size (bebas), 120 = n_steps, 1 = fitur
                        layer["config"]["batch_shape"] = (None, 120, 1)
                    
                    # 2. Hapus parameter yang tidak dikenal Keras 3
                    layer["config"].pop("time_major", None)
                    layer["config"].pop("batch_input_shape", None)

        # Daftar objek yang harus dikenali
        custom_objects = {
            "Sequential": keras.models.Sequential,
            "LSTM": keras.layers.LSTM,
            "Dense": keras.layers.Dense,
            "InputLayer": keras.layers.InputLayer,
            "Dropout": keras.layers.Dropout
        }

        # Rakit model
        model = model_from_json(json.dumps(config_dict), custom_objects=custom_objects)
        
        # Load weights
        model.load_weights(model_path)
        return model

# Streamlit app
def main():
    st.title("Stock Price Prediction Web App")

    # Sidebar for data download
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., BBCA.JK):", "BBCA.JK")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    # Tambahkan pengecekan ini agar tidak error
    if data.empty:
        st.error(f"Data untuk simbol {stock_symbol} tidak ditemukan atau kosong. Pastikan simbol benar (contoh: BBCA.JK) dan rentang tanggal sesuai.")
        return # Menghentikan proses agar tidak lanjut ke scaler yang kosong
    data = data.dropna()

    # Data preprocessing
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Data preparation
    n_steps = 120
    X, y = prepare_data(scaled_data, n_steps)

    # Splitting into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape data for LSTM and GRU models
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Sidebar for model selection
    st.sidebar.header("Select Model")
    model_type = st.sidebar.selectbox("Select Model Type:", ["LSTM", "GRU"])

    # Load saved models
    if model_type == "LSTM":
        final_model = load_legacy_model("final_model_lstm.h5")
    else:
        final_model = load_legacy_model("final_model_gru.h5")

    # Model evaluation
    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # TAMBAHKAN INI: Pastikan tidak ada NaN sebelum hitung MSE
    mask = ~np.isnan(y_test_orig) & ~np.isnan(y_pred)
    y_test_orig = y_test_orig[mask]
    y_pred = y_pred[mask]

    # Baris 113 yang tadinya error sekarang akan aman
    mse = mean_squared_error(y_test_orig, y_pred)



    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    # Display results
    st.header(f"Results for {model_type} Model")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("Mean Absolute Percentage Error (MAPE):", mape)

    # Visualize predictions
    st.header("Visualize Predictions")
    visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred)

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # Ambil data dari index i sampai i + n_steps sebagai input (X)
        X.append(data[i:(i + n_steps), 0])
        # Ambil data tepat setelah n_steps sebagai target (y)
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig.update_layout(title="Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (IDR)",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()