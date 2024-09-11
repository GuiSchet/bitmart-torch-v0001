import ccxt
import time
import math
from datetime import datetime
from sre_constants import error
import pandas as pd
import numpy as np
import sqlite3
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sre_constants import error

load_dotenv()

## Inicializacion del exchange.

exchange_id = 'bitmart'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': os.getenv('API_KEY'),
    'secret': os.getenv('API_SECRET'),
    'timeout': 30000,
    'enableRateLimit': True,
})
markets = exchange.load_markets ()
delay = 2 # seconds
markets_list = []
mercadoBase = 'BTC'

## Iterar por todas las monedas para dejar solo mercadoBase
for key in markets.keys():
  
  # Check if '/BTC' is in the key and if 'state' exists in the 'info' dictionary and is 'online'
  if mercadoBase == markets[key]['quote']:
    markets_list.append(key)

#print(exchange.has)


## Carga OHLCV.
def dataOHLCV(market):
  try:
    # Calculate the timestamp for 100 days ago
    since = int((time.time() - 199 * 60 * 60) * 1000)
    ohlcv_data = exchange.fetchOHLCV(market, '1h', since, 1000)


    # Create a Pandas DataFrame from the OHLCV data
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df
  except:
    print(datetime.now(), '/// Error cargando data desde OHLCV1.')

## Analisis con torch.
# Definir el modelo LSTM
# Definimos la clase del modelo LSTM
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

# Función para crear secuencias de datos
def crear_secuencias(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-1])  # Usamos todas las columnas excepto la de labels ('close')
        y.append(data[i + n_steps, -1])  # El label es el 'close'
    return np.array(X), np.array(y)

# Función principal para la predicción y entrenamiento
def prediccion6h(market):
    data = dataOHLCV(market)  # Aquí obtienes los datos de OHLCV (open, high, low, close, volume)

    # Asegúrate de que 'timestamp' es el índice del DataFrame
    data = data.dropna()  # Elimina cualquier fila con valores faltantes

    # Features y labels
    features = data[['open', 'high', 'low', 'volume']].values
    labels = data['close'].values.reshape(-1, 1)  # Reshape para tener la misma dimensión que las features

    # Preprocesamiento
    scaler_features = MinMaxScaler()
    scaler_labels = MinMaxScaler()

    features_scaled = scaler_features.fit_transform(features)
    labels_scaled = scaler_labels.fit_transform(labels)

    # Combina features y labels para usar en la predicción
    dataset_scaled = np.hstack((features_scaled, labels_scaled))

    # Crear secuencias de entrada (usando las últimas 6 horas para predecir la siguiente)
    n_steps = 6  # Usaremos las últimas 6 horas como input para predecir
    X, y = crear_secuencias(dataset_scaled, n_steps)

    # Convertimos las secuencias a tensores de PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Agregar una dimensión extra para el tensor

    # Definir el modelo LSTM
    input_size = X.shape[2]  # El número de features
    hidden_size = 50
    output_size = 1
    model = LSTMModel(input_size, hidden_size, output_size)
    
    # Cargar el modelo en el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Definir hiperparámetros
    num_epochs = 40  # Número de epochs que quieres
    learning_rate = 0.001
    criterion = nn.MSELoss()  # Definir función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Bucle de entrenamiento
    model.train()  # Cambiar a modo de entrenamiento
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Limpiar gradientes
        outputs = model(X_tensor.to(device))  # Hacer forward
        loss = criterion(outputs, y_tensor.to(device))  # Calcular la pérdida
        loss.backward()  # Hacer backward (propagación hacia atrás)
        optimizer.step()  # Actualizar los parámetros

        if (epoch+1) % 10 == 0:  # Imprimir cada 10 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Modo evaluación para hacer predicciones
    model.eval()

    # Usar los últimos n_steps del dataset como input para predecir las próximas 6 horas de forma secuencial
    input_data = dataset_scaled[-n_steps:, :-1]  # Usamos las últimas 6 horas de features
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, num_features)

    predicciones = []

    with torch.no_grad():
        for _ in range(6):  # Predecir las próximas 6 horas
            pred_scaled = model(input_tensor).cpu().numpy()  # Obtenemos la predicción escalada
            pred_descaled = scaler_labels.inverse_transform(pred_scaled)  # Invertimos el escalado
            
            # Guardar la predicción desescalada
            predicciones.append(pred_descaled.flatten()[0])

            # Crear un nuevo input concatenando las últimas 5 horas de input con la nueva predicción
            nueva_prediccion_features = np.array([[pred_scaled[0][0], pred_scaled[0][0], pred_scaled[0][0], 0]])  # Suponiendo que open, high, low sean iguales, y volume = 0
            nuevo_input = np.vstack((input_tensor.cpu().numpy()[0, 1:, :], nueva_prediccion_features))  # Elimina el primer valor y añade la predicción
            input_tensor = torch.tensor(nuevo_input, dtype=torch.float32).unsqueeze(0).to(device)  # Actualiza el input_tensor para la siguiente hora

    return predicciones, data['close'].iloc[-1]


if __name__ == '__main__':
    markets_up = []
    for market in markets_list:
        time.sleep(2)
        try:
            pred, price_close = prediccion6h(market)
            pred_mean = np.mean(pred)
        except:
            print(f"Error tomando predicciones en {market}")
            continue
        if pred_mean > price_close and pred_mean / price_close < 1.4 and pred_mean / price_close > 1.02:
            markets_up.append([market, pred_mean, price_close, pred_mean / price_close])
    markets_up = sorted(markets_up, key=lambda x: x[3], reverse=True)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    for x in markets_up:
        print(x)



