import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import ta

data = pd.read_csv("./raw/ETHUSD.csv")
data['Date'] = data['Date'].astype(str)
data['Timestamp'] = data['Timestamp'].astype(str)

data['DateTime'] = data['Date'] + data['Timestamp']
del data['Date']
del data['Timestamp']

data['DateTime'] = pd.to_datetime(data['DateTime'], format="%Y%m%d%H:%M:%S")
data = data.set_index('DateTime')

#data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()

data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3

# Calculate 20-Period EMA based on Typical Price
#data['ema_20'] = data['typical_price'].ewm(span=20, adjust=False).mean()

#print(data['ema_20'].autocorr(lag=1))
print(data['Close'].autocorr(lag=18))

def add_indicators(df):
    
    #Trend Indicators
    df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()

    # Momentum Indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()  # Relative Strength Index
    df['macd'] = ta.trend.MACD(close=df['Close']).macd()                      # MACD Line
    df['macd_signal'] = ta.trend.MACD(close=df['Close']).macd_signal()        # MACD Signal Line

    # Moving Averages
    df['ema_9'] = ta.trend.EMAIndicator(close=df['Close'], window=9).ema_indicator()  # Simple Moving Average
    df['ema_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()  # Exponential Moving Average
    df['ema_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()  # Exponential Moving Average
    df['ema_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()  # Exponential Moving Average

    # Volatility Indicators
    df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()  # ATR
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()   # Bollinger Upper Band
    df['bb_lower'] = bb.bollinger_lband()   # Bollinger Lower Band

    # Volume Indicators
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()  # OBV

    return df

data = add_indicators(data)
data = data.dropna()

n_steps = 54  # Number of past candlesticks to use
future_steps = 18*3  # Number of steps in the future for prediction (30 minutes)

#data['Target'] = data['Close'].shift(-future_steps)
#data = data.iloc[33:-10]
#print(data.isnull().sum())

# Create sequences
def create_sequences(data, n_steps):
    X = []
    atr = []
    y = []
    for i in range(len(data) - n_steps - future_steps):
        X.append(data['Close'][i:(i + n_steps)].values)
        atr.append(data.iloc[i + n_steps]['ATR'])
        y.append(data.iloc[i + n_steps + future_steps]['Close'])
    return np.array(X), np.array(atr), np.array(y)

# Create sequences
#X, atr, y = create_sequences(data[['ema_20', 'Close', 'ATR']], n_steps)

# Create a new dataframe with the sequences and target
#sequence_df = pd.DataFrame({
#    'sequence': [x for x in X],
#    'ATR': atr,
#    'target': y
#})

#print(sequence_df.head())
#print(sequence_df['sequence'].shape)

#data['Target'] = data['Close'].shift(-future_steps)
train, test = train_test_split(data, test_size=0.2, shuffle=False)

#data.to_csv('./processed/ethusd.csv')
train.to_pickle('./processed/train.pkl')
test.to_pickle('./processed/test.pkl')

print(train.dtypes)
print(train.head(11))
print(len(train))
