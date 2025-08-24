import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import ta
import backtrader as bt

import torch

from model.mlp import MLP

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Create a subclass of Strategy to define the indicators and logic
class SuperTrend(bt.Indicator):
    """
    SuperTrend Algorithm :
    BASIC UPPERBAND = (high + low) / 2 + Multiplier * ATR
    BASIC lowERBAND = (high + low) / 2 - Multiplier * ATR
    FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous close > Previous FINAL UPPERBAND))
                        THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
    FINAL lowERBAND = IF( (Current BASIC lowERBAND > Previous FINAL lowERBAND) or (Previous close < Previous FINAL lowERBAND))
                        THEN (Current BASIC lowERBAND) ELSE Previous FINAL lowERBAND)
    SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current close <= Current FINAL UPPERBAND)) THEN
                    Current FINAL UPPERBAND
                ELSE
                    IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current close > Current FINAL UPPERBAND)) THEN
                        Current FINAL lowERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL lowERBAND) and (Current close >= Current FINAL lowERBAND)) THEN
                            Current FINAL lowERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL lowERBAND) and (Current close < Current FINAL lowERBAND)) THEN
                                Current FINAL UPPERBAND
    """
    lines = ('super_trend',)
    params = (('period', 10),
              ('multiplier', 2),
              )
    plotlines = dict(
        super_trend=dict(
            _name='ST',
            color='blue',
            alpha=1
        )
    )
    plotinfo = dict(subplot=False)

    def __init__(self):
        self.st = [0]
        self.finalupband = [0]
        self.finallowband = [0]
        self.addminperiod(self.p.period)
        atr = bt.ind.ATR(self.data, period=self.p.period)
        self.upperband = (self.data.high + self.data.low) / 2 + self.p.multiplier * atr
        self.lowerband = (self.data.high + self.data.low) / 2 - self.p.multiplier * atr

    def next(self):
        pre_upband = self.finalupband[0]
        pre_lowband = self.finallowband[0]
        if self.upperband[0] < self.finalupband[-1] or self.data.close[-1] > self.finalupband[-1]:
            self.finalupband[0] = self.upperband[0]
        else:
            self.finalupband[0] = self.finalupband[-1]
        if self.lowerband[0] > self.finallowband[-1] or self.data.close[-1] < self.finallowband[-1]:
            self.finallowband[0] = self.lowerband[0]
        else:
            self.finallowband[0] = self.finallowband[-1]
        if self.data.close[0] <= self.finalupband[0] and ((self.st[-1] == pre_upband)):
            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.finalupband[0]
        elif (self.st[-1] == pre_upband) and (self.data.close[0] > self.finalupband[0]):
            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]
        elif (self.st[-1] == pre_lowband) and (self.data.close[0] >= self.finallowband[0]):
            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]
        elif (self.st[-1] == pre_lowband) and (self.data.close[0] < self.finallowband[0]):
            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.st[0]


class SuperTrendStrategy(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


def add_indicators(df):
    # Momentum Indicators
    #df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()  # Relative Strength Index
    #df['macd'] = ta.trend.MACD(close=df['Close']).macd()                      # MACD Line
    #df['macd_signal'] = ta.trend.MACD(close=df['Close']).macd_signal()        # MACD Signal Line
    # Feature engineering for MACD
    #df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, 0)


    # Moving Averages
    #df['sma_9'] = ta.trend.SMAIndicator(close=df['Close'], window=9).sma_indicator()  # Simple Moving Average
    #df['ema_21'] = ta.trend.EMAIndicator(close=df['Close'], window=21).ema_indicator()  # Exponential Moving Average

    # Volatility Indicators
    #df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()  # ATR
    #bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    #df['bb_upper'] = bb.bollinger_hband()   # Bollinger Upper Band
    #df['bb_lower'] = bb.bollinger_lband()   # Bollinger Lower Band

    # Volume Indicators
    #df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()  # OBV
    df['supertrend'] = df.ta.supertrend(period=10, multiplier=1, append=True)
    print(df['supertrend'])

    return df


test = pd.read_pickle('./data/processed/test.pkl')
test = add_indicators(test)

test = test.dropna()

backtest_set = test[['Open', 'High', 'Low', 'Close', 'Volume']]

cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

#Create a data feed
data = bt.feeds.PandasData(dataname=backtest_set)

cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(SuperTrend)  # Add the trading strategy
results = cerebro.run()  # run it all
cerebro.plot()  # and plot it with a single command
