import ccxt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

## Inicializacion del exchange.

exchange_id = 'bitmart'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': os.getenv('BITMART_API_KEY'),
    'secret': os.getenv('BITMART_SECRET'),
    'timeout': 30000,
    'enableRateLimit': True,
})
markets = exchange.load_markets()
delay = 2  # seconds
mercadoBase = 'BTC'
markets_list = []
for key in markets.keys():
    if mercadoBase == markets[key]['quote']:
        markets_list.append(key)

markets_list = ['LTC/BTC', 'ZRX/BTC', 'ATOM/BTC', 'XLM/BTC', 'TRX/BTC', 'TRAC/BTC', 'DASH/BTC', 'LINK/BTC', 'XMR/BTC', 'SOL/BTC', 'BNB/BTC', 'MKR/BTC', 'ZIL/BTC', 'QTUM/BTC', 'VET/BTC', 'ONT/BTC', 'BCH/BTC', 'ETH/BTC']

## Carga OHLCV.
def dataOHLCV(market):
    try:
        since = int((time.time() - 199 * 60 * 60) * 1000)
        ohlcv_data = exchange.fetchOHLCV(market, '1h', since, 1000)
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error cargando data desde OHLCV para {market}: {e}")
        return None

# Implementación manual de indicadores técnicos.

# RSI (Relative Strength Index)
def calculate_RSI(data, window=14):
    delta = data['close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD (Moving Average Convergence Divergence)
def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    
    return macd, signal

# Medias móviles simples (SMA)
def calculate_SMA(data, window):
    return data['close'].rolling(window=window).mean()

# Función para crear secuencias de datos
def crear_secuencias(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-1])  # Usamos todas las columnas excepto la de labels ('close')
        y.append(data[i + n_steps, -1])  # El label es el 'close'
    return np.array(X), np.array(y)

# Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Tomamos solo la última salida de la secuencia
        return out

# Verificación de NaN o Inf en los datos antes de escalar
def check_nan_inf(data):
    if np.isnan(data).any() or np.isinf(data).any():
        print("Datos contienen NaN o Inf.")
    else:
        print("Datos verificados, no contienen NaN ni Inf.")

# Modificar la función de predicción para incluir verificaciones de datos
def prediccion6h(market):
    try:
        data = dataOHLCV(market)  
        if data is None or len(data) == 0:
            print(f"Datos insuficientes para {market}.")
            return None, None

        print(f"Datos originales para {market}: {len(data)}")

        # Calcular indicadores técnicos con periodos más cortos
        data['RSI'] = calculate_RSI(data, window=9)
        data['MACD'], data['Signal'] = calculate_MACD(data, short_window=6, long_window=13, signal_window=5)
        data['SMA_50'] = calculate_SMA(data, window=20)
        data['SMA_200'] = calculate_SMA(data, window=50)

        # Eliminar o rellenar filas con NaN
        data.ffill(inplace=True)  # Usamos forward-fill para evitar errores con NaN
        data.bfill(inplace=True)  # Back-fill en caso de que sea necesario

        print(f"Datos después del cálculo de indicadores para {market}: {len(data)}")
        
        # Verificar si hay suficientes datos después de los cálculos
        if len(data) < 12:
            print(f"No hay suficientes datos después del cálculo de indicadores para {market}.")
            return None, None

        # Features y labels
        features = data[['open', 'high', 'low', 'volume', 'RSI', 'MACD', 'Signal', 'SMA_50', 'SMA_200']].values
        labels = data['close'].values.reshape(-1, 1)

        # Verificar si hay NaN o Inf en los datos antes de escalar
        check_nan_inf(features)
        check_nan_inf(labels)

        # Preprocesamiento
        scaler_features = MinMaxScaler()
        scaler_labels = MinMaxScaler()

        features_scaled = scaler_features.fit_transform(features)
        labels_scaled = scaler_labels.fit_transform(labels)

        # Verificar después de escalar
        check_nan_inf(features_scaled)
        check_nan_inf(labels_scaled)

        dataset_scaled = np.hstack((features_scaled, labels_scaled))

        # Crear secuencias de entrada
        n_steps = 6
        X, y = crear_secuencias(dataset_scaled, n_steps)

        if len(X) == 0:
            print(f"No se pudieron generar secuencias suficientes para {market}.")
            return None, None

        # Convertimos las secuencias a tensores de PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Definir el modelo LSTM
        input_size = X.shape[2]
        hidden_size = 50
        output_size = 1
        model = LSTMModel(input_size, hidden_size, output_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Inicializar pesos del modelo
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.01)

        model.apply(init_weights)

        # Hiperparámetros
        num_epochs = 40
        learning_rate = 0.001
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor.to(device))
            loss = criterion(outputs, y_tensor.to(device))
            
            # Verificar si la pérdida es NaN antes de continuar
            if torch.isnan(loss).any():
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: nan - Datos incorrectos detectados.")
                break

            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        model.eval()
        input_data = dataset_scaled[-n_steps:, :-1]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

        predicciones = []
        with torch.no_grad():
            for _ in range(6):
                pred_scaled = model(input_tensor).cpu().numpy()
                pred_descaled = scaler_labels.inverse_transform(pred_scaled)
                predicciones.append(pred_descaled.flatten()[0])

                # Crear la nueva predicción con todas las features necesarias
                nueva_prediccion_features = np.array([[pred_scaled[0][0], pred_scaled[0][0], pred_scaled[0][0], input_data[-1, 3],  # volumen
                                                       input_data[-1, 4],  # RSI
                                                       input_data[-1, 5],  # MACD
                                                       input_data[-1, 6],  # Signal
                                                       input_data[-1, 7],  # SMA_50
                                                       input_data[-1, 8]]])  # SMA_200

                # Actualizamos el input para la siguiente predicción
                nuevo_input = np.vstack((input_tensor.cpu().numpy()[0, 1:, :], nueva_prediccion_features))
                input_tensor = torch.tensor(nuevo_input, dtype=torch.float32).unsqueeze(0).to(device)

        return predicciones, data['close'].iloc[-1]

    except Exception as e:
        print(f"Error en prediccion6h para {market}: {e}")
        return None, None


if __name__ == '__main__':
    markets_up = []
    for market in markets_list:
        time.sleep(2)
        try:
            pred, price_close = prediccion6h(market)
            if pred is None:
                continue
            pred_mean = np.mean(pred)
        except Exception as e:
            print(f"Error tomando predicciones en {market}: {e}")
            continue
        if pred_mean > price_close and pred_mean / price_close > 1.02:
            markets_up.append([market, pred_mean, price_close, pred_mean / price_close])
    markets_up = sorted(markets_up, key=lambda x: x[3], reverse=True)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    for x in markets_up:
        print(x)

