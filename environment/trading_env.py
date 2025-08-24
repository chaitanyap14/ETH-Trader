import numpy as np

class TradingEnv:
    def __init__(self, data, lookback_window=1, risk_percent=0.01):
        self.data = data
        self.lookback_window = lookback_window
        self.current_step = self.lookback_window
        self.acount_balance = 10000
        self.risk_percent = risk_percent
        self.sl_multiplier = 1
        self.tp_multiplier = 2.5
        self.position = {
            'position': None,
            'entry_price': None,
            'sl': None,
            'tp': None
        }
        self.win_streak = 0
        self.loss_streak = 0
        self.opening_action = None
        self.opening_state = None
        self.reset()
        
    def reset(self):
        self.current_step = self.lookback_window
        self.account_balance = 10000  # Initial balance
        self.position['position'] = None
        self.position['entry_price'] = None
        self.position['sl'] = None
        self.position['tp'] = None
        self.win_streak = 0
        self.loss_streak = 0
        return self._get_state()

    def _get_state(self):
        state = self.data[self.current_step-self.lookback_window:self.current_step]
        market_state = np.array([
            #state['Open'],
            #state['High'],
            #state['Low'],
            #state['Close'],
            #state['Volume'],
            #state['typical_price'],
            #state['ema_20'],
            #state['sma_9'],
            #state['rsi'],
            #state['macd'],
            #state['macd_signal'],
            #state['adx'],
            #state['atr'],
            #state['obv'],
            #state['bb_lower'],
            #state['bb_upper']

            (state['Open'].values > state['Close'].shift(1).values).astype(int),       # 1 if open > prev close
            (state['Close'].values > state['Close'].shift(1).values).astype(int),       # 1 if close > prev close
            (state['Close'].shift(1).values > state['Close'].shift(2).values).astype(int),       # 1 if close > prev close
            (state['Close'].shift(2).values > state['Close'].shift(3).values).astype(int),       # 1 if close > prev close
            (state['Close'].shift(3).values > state['Close'].shift(4).values).astype(int),       # 1 if close > prev close
            (state['Close'].shift(4).values > state['Close'].shift(5).values).astype(int),       # 1 if close > prev close
            (state['Close'].values > state['ema_9'].values).astype(int),               # 1 if close > SMA9
            (state['Close'].values > state['ema_20'].values).astype(int),               # 1 if high > EMA20
            (state['Close'].values > state['ema_50'].values).astype(int),               # 1 if high > EMA20
            (state['Close'].values > state['ema_200'].values).astype(int),               # 1 if high > EMA20
            (state['Volume'].values > state['Volume'].rolling(20).mean().values).astype(int),
            (state['typical_price'].values > state['typical_price'].mean()).astype(int),
            (state['rsi'].values > 50).astype(int),                                    # 1 if RSI > 50
            (state['rsi'].values > 70).astype(int),                                    # 1 if RSI > 50
            (state['rsi'].values > 30).astype(int),                                    # 1 if RSI > 50
            (state['macd'].values > state['macd_signal'].values).astype(int),          # 1 if MACD > signal
            (state['adx'].values > 25).astype(int),                                    # 1 if ADX > 25
            (state['adx'].values > 50).astype(int),                                    # 1 if ADX > 25
            (state['adx'].values > 75).astype(int),                                    # 1 if ADX > 25
            (state['atr'].values > state['atr'].mean()).astype(int),                   # 1 if ATR > mean
            (state['obv'].values > state['obv'].rolling(20).mean().values).astype(int),# 1 if OBV > 20MA
            (state['Close'].values < state['bb_lower'].values).astype(int),            # 1 if close < BB lower
            (state['Close'].values > state['bb_upper'].values).astype(int)             # 1 if close > BB upper
        ])

        return market_state


    def step(self, action):
        done = self.current_step >= len(self.data) - 1
        reward = 0
        executed_action = None
        executed_state = None
        
        if self.account_balance > 0 and self.account_balance <= 10000000000:
            if self.position['position'] is not None:
                reward, executed_action, executed_state = self._handle_position()
            else:
                self._open_position(action)
                executed_action = action
                executed_state = self._get_state()
            
        self.current_step += 1
        next_state = self._get_state()

        return next_state, reward, done, executed_state, executed_action


    def _open_position(self, action):
        # Implement position opening logic with desired risk-reward
        atr = self.data['atr'].iloc[self.current_step]
        if action == 0:
            self.position['position'] = 1
            self.position['entry_price'] = self.data['Close'].iloc[self.current_step]
            self.position['sl'] = self.position['entry_price'] - (atr * self.sl_multiplier)
            self.position['tp'] = self.position['entry_price'] + (atr * self.tp_multiplier)
            self.opening_action = action
            self.opening_state = self._get_state()

        elif action == 1:
            self.position['position'] = -1
            self.position['entry_price'] = self.data['Close'].iloc[self.current_step]
            self.position['sl'] = self.position['entry_price'] + (atr * self.sl_multiplier)
            self.position['tp'] = self.position['entry_price'] - (atr * self.tp_multiplier)
            self.opening_action = action
            self.opening_state = self._get_state()

        else:
            pass
            

    def old_handle_position(self):
        # Implement position management logic
        spread = 0.005*self.data['atr'].iloc[self.current_step]
        reward = 0
        
        if self.position['position'] == 1:
            if self.data['Low'].iloc[self.current_step] <= self.position['sl']:
                reward = self.account_balance * (1 - self.risk_percent) - self.account_balance
                #reward = -1
                self.position['position'] = None
                self.position['entry_price'] = None
                self.position['sl'] = None
                self.position['tp'] = None
            elif self.data['High'].iloc[self.current_step] >= self.position['tp']:
                reward = self.account_balance * (1 + (self.tp_multiplier/self.sl_multiplier) * self.risk_percent) - self.account_balance
                #reward = 3
                self.position['position'] = None
                self.position['entry_price'] = None
                self.position['sl'] = None
                self.position['tp'] = None
            else:
                pass

        elif self.position['position'] == -1:
            if self.data['High'].iloc[self.current_step] + spread >= self.position['sl']:
                reward = self.account_balance * (1 - self.risk_percent) - self.account_balance
                #reward = -1
                self.position['position'] = None
                self.position['entry_price'] = None
                self.position['sl'] = None
                self.position['tp'] = None
            elif self.data['Low'].iloc[self.current_step] + spread <= self.position['tp']:
                reward = self.account_balance * (1 + (self.tp_multiplier/self.sl_multiplier) * self.risk_percent) - self.account_balance
                #reward = 3
                self.position['position'] = None
                self.position['entry_price'] = None
                self.position['sl'] = None
                self.position['tp'] = None
            else:
                pass

        else:
            pass

        self.account_balance += reward

        if reward > 0:
            reward = 3
        elif reward < 0:
            reward = -1
        else:
            reward = 0

        return reward



    def _handle_position(self):
        spread = 0.005*self.data['atr'].iloc[self.current_step]
        reward = 0
        is_win = False
        is_loss = False

        if self.position['position'] == 1:
            if self.data['Low'].iloc[self.current_step] <= self.position['sl']:
                is_loss = True
            elif self.data['High'].iloc[self.current_step] >= self.position['tp']:
                is_win = True

        elif self.position['position'] == -1:
            if self.data['High'].iloc[self.current_step] + spread >= self.position['sl']:
                is_loss = True
            elif self.data['Low'].iloc[self.current_step] + spread <= self.position['tp']:
                is_win = True

        # Update streaks
        if is_win:
            self.account_balance *= 1+self.risk_percent*(self.tp_multiplier/self.sl_multiplier)
            self.win_streak += 1
            self.loss_streak = 0
            reward = 3 #* (1 + 0.1 * min(self.win_streak, 5))  # Max 50% bonus
        elif is_loss:
            self.account_balance *= 1-self.risk_percent
            self.loss_streak += 1
            self.win_streak = 0
            reward = -1 #* (1 + 0.1 * min(self.loss_streak, 3))  # Max 30% penalty
        else:
            reward = 0

        # Clean up position
        if is_win or is_loss:
            action = self.opening_action
            state = self.opening_state
            self.opening_action = None
            self.opening_state = None
            self.position = {'position': None, 'entry_price': None, 'sl': None, 'tp': None}
        else:
            reward = 0
            action = None
            state = None

        return reward, action, state


