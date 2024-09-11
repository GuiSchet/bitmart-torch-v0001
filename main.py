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
mercadoBase = 'USDT'

print(markets)