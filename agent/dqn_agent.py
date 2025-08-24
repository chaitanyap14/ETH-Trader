#import numpy as np
#import tensorflow as tf
#from collections import deque
#import random
#import heapq
#
#class DQNAgent:
#    def __init__(self, state_size, action_size):
#        self.state_size = state_size
#        self.action_size = action_size
#        self.gamma = 0.95
#        self.epsilon = 1.0
#        self.epsilon_min = 0.05
#        self.epsilon_decay = 0.9995
#        self.batch_size = 512
#        self.lookback_window = 12
#        self.model = self._build_model()
#        self.memory = deque(maxlen=100000)
#
#
#    def _build_model(self):
#        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0000003, rho=0.95, epsilon=1e-7, clipnorm=1.0)
#        model = tf.keras.Sequential([
#            tf.keras.layers.Input(shape=(self.state_size, 16)),
#            #tf.keras.layers.Reshape((self.state_size, 16, 1)),
#            #tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')),
#            #tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
#            #tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
#            #tf.keras.layers.Flatten(),
#            tf.keras.layers.LayerNormalization(),
#            tf.keras.layers.LSTM(512, dropout=0.2),  # 16 features
#            tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#            tf.keras.layers.Dense(self.action_size, activation='linear')
#        ])
#        model.compile(loss='mse', optimizer=optimizer)
#        return model
#
#    def create_dataset(self):
#        def generator():
#            minibatch = random.sample(self.memory, self.batch_size)
#            states = np.array([t[0] for t in minibatch])
#            actions = np.array([t[1] for t in minibatch])
#            rewards = np.array([t[2] for t in minibatch])
#            next_states = np.array([t[3] for t in minibatch])
#            dones = np.array([t[4] for t in minibatch])
#
#            yield (
#                states.reshape(self.batch_size, self.lookback_window, 16),
#                actions,
#                rewards,
#                next_states.reshape(self.batch_size, self.lookback_window, 16),
#                dones.astype(float),
#            )
#
#        self.dataset = tf.data.Dataset.from_generator(
#            generator,
#            output_signature=(
#                tf.TensorSpec(shape=(self.batch_size, self.lookback_window, 16), dtype=tf.float32),
#                tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
#                tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32),
#                tf.TensorSpec(shape=(self.batch_size, self.lookback_window, 16), dtype=tf.float32),
#                tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32),
#            )
#        ).repeat().prefetch(tf.data.AUTOTUNE)
#
#    def remember(self, state, action, reward, next_state, done):
#        self.memory.append((state, action, reward, next_state, done))
#
#
#
#    def oldact(self, state):
#        if np.random.rand() <= self.epsilon:
#            return random.randrange(self.action_size)
#        state = state.reshape(1, self.lookback_window, 16)
#        act_values = self.model.predict(state, verbose=0)
#        return np.argmax(act_values[0])
#
#    @tf.function
#    def act(self, state):
#        if tf.random.uniform(()) <= self.epsilon:
#            return tf.cast(tf.random.uniform((), minval=0, maxval=self.action_size, dtype=tf.int32), tf.int64)
#        state = tf.reshape(state, (1, self.lookback_window, 16))
#        act_values = self.model(state, training=False)
#        return tf.cast(tf.argmax(act_values[0]), tf.int64)
#
#
#
#    def oldreplay(self):
#        if len(self.memory) < self.batch_size:
#            return
#        
#        minibatch = random.sample(self.memory, self.batch_size)
#        states = np.array([t[0] for t in minibatch])
#        next_states = np.array([t[3] for t in minibatch])
#
#        states = states.reshape(self.batch_size, self.lookback_window, 16)
#        next_states = next_states.reshape(self.batch_size, self.lookback_window, 16)
#
#        targets = self.model.predict(states, verbose=0)
#        next_q_values = self.model.predict(next_states, verbose=0)
#
#        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
#            if done:
#                targets[i][action] = reward
#            else:
#                targets[i][action] = reward + self.gamma * np.amax(next_q_values[i])
#
#        self.model.fit(states, targets, epochs=1, verbose=0)
#
#
#    @tf.function()
#    def train_step(self, states, actions, rewards, next_states, dones):
#        with tf.GradientTape() as tape:
#            # Predict Q-values for current and next states
#            q_values = self.model(states)
#            next_q_values = self.model(next_states)
#    
#            # Calculate targets
#            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
#            updates = rewards + self.gamma * max_next_q_values * (1 - dones)
#    
#            # Create indices for scatter update
#            indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
#    
#            # Update targets
#            targets = tf.tensor_scatter_nd_update(q_values, indices, updates)
#    
#            # Compute loss
#            loss = tf.reduce_mean(tf.square(targets - q_values))
#    
#        # Apply gradients
#        grads = tape.gradient(loss, self.model.trainable_variables)
#        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
#        return loss
#
#
#    def replay(self):
#        if len(self.memory) < self.batch_size:
#            return
#
#        # Sample a minibatch from memory
#        minibatch = random.sample(self.memory, self.batch_size)
#        states = np.array([t[0] for t in minibatch])
#        actions = np.array([t[1] for t in minibatch])
#        rewards = np.array([t[2] for t in minibatch])
#        next_states = np.array([t[3] for t in minibatch])
#        dones = np.array([t[4] for t in minibatch])
#
#        # Reshape states and next_states to match the model input shape
#        states = states.reshape(self.batch_size, self.lookback_window, 16)
#        next_states = next_states.reshape(self.batch_size, self.lookback_window, 16)
#
#        # Convert everything to tensors
#        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
#        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
#        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
#        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
#        dones_tensor = tf.convert_to_tensor(dones.astype(float), dtype=tf.float32)
#
#        # Call the compiled train_step function
#        loss = self.train_step(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
#
#        # Decay epsilon
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
#
#    @tf.function
#    def batch_act(self, states):
#        """Process multiple states at once"""
#        return tf.argmax(self.model(states), axis=1)
#    
#    def act_batch(self, states):
#        """Vectorized epsilon-greedy"""
#        rand_mask = np.random.rand(len(states)) < self.epsilon
#        random_actions = np.random.randint(0, self.action_size, size=len(states))
#        model_actions = self.batch_act(np.stack(states)).numpy()
#        return np.where(rand_mask, random_actions, model_actions)

import numpy as np
import tensorflow as tf
from lightgbm import LGBMClassifier
from collections import defaultdict, deque
import random
import joblib

from tqdm import tqdm

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.batch_size = 512
        self.lookback_window = 12

        # Simplified PER parameters
        self.alpha = 0.7  # Balanced prioritization
        self.beta = 0.5   # Moderate importance sampling
        self.abs_err_upper = 5.0  # Slightly higher ceiling

        self.memory = deque(maxlen=100000)
        self.priorities = deque(maxlen=100000)

        # Single model (no target network for speed)
        self.model = self._build_model()

        # Track best rewards to adjust learning
        self.best_reward = -np.inf

    def _build_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size, 16)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='linear'),
            tf.keras.layers.Dense(128, activation='linear'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='huber_loss', optimizer=optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        # Boost priority for high rewards
        priority = (abs(reward) + 1e-5) ** 1.5  # Non-linear boost
        if action is not None and state is not None:
            """Store experience with volatility-based sampling"""
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(min(priority, self.abs_err_upper))

    def sample(self):
        # Fast sampling using numpy
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return indices, weights

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, weights):
        with tf.GradientTape() as tape:
            # Standard DQN (no target network)
            next_q_values = self.model(next_states)
            targets = rewards + self.gamma * tf.reduce_max(next_q_values, axis=1) * (1 - dones)
            targets = tf.clip_by_value(targets,
                              clip_value_min=-20,  # -1/(1-γ)
                              clip_value_max=60)   # 3/(1-γ)
            #tf.print(targets)

            current_q = tf.gather_nd(
                self.model(states),
                tf.stack([tf.range(self.batch_size), tf.cast(actions, tf.int32)], axis=1)
            )

            td_errors = tf.abs(targets - current_q)
            loss = tf.reduce_mean(weights * tf.where(
                td_errors < 1.0,
                0.5 * tf.square(td_errors),
                td_errors - 0.5
            ))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return td_errors

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Fast sampling and training
        indices, weights = self.sample()
        batch = [self.memory[i] for i in indices]

        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Convert to tensors
        states_t = tf.convert_to_tensor(states.reshape(-1, self.lookback_window, 16), dtype=tf.float32)
        next_t = tf.convert_to_tensor(next_states.reshape(-1, self.lookback_window, 16), dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_t = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Train
        td_errors = self.train_step(states_t, actions_t, rewards_t, next_t, dones_t, weights_t)

        # Update priorities
        td_errors = td_errors.numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = min(td_errors[i] + 1e-6, self.abs_err_upper)

        # Adaptive epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @tf.function
    def oldact(self, state):
        if tf.random.uniform(()) <= self.epsilon:
            return tf.random.uniform((), 0, self.action_size, dtype=tf.int32)
        state = tf.reshape(state, (1, self.lookback_window, 16))
        return tf.argmax(self.model(state)[0])

    @tf.function
    def act(self, state):
        # Generate random action with same dtype as model output
        random_action = tf.cast(
            tf.random.uniform((), minval=0, maxval=self.action_size, dtype=tf.int32),
            tf.int64
        )
    
        # Model-predicted action
        deterministic_action = tf.cast(
            tf.argmax(self.model(tf.reshape(state, (1, self.lookback_window, 16)))[0]),
            tf.int64
        )
    
        # Epsilon-greedy selection
        return tf.cond(
            tf.random.uniform(()) <= self.epsilon,
            lambda: random_action,
            lambda: deterministic_action
        )


    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)


class DTAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 256
        
        # Initialize with explicit num_class parameter
        self.model = LGBMClassifier(
            n_estimators=80,
            max_depth=5,
            num_leaves=20,
            min_data_in_leaf=10,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            learning_rate=0.05,
            objective='multiclass',
            num_class=action_size,  # CRITICAL FIX
            metric='multi_logloss',
            verbosity=-1,
            n_jobs=-1,
            force_row_wise=True
        )
        
        self.state_buffer = np.zeros((self.batch_size, state_size * 16))
        self.action_buffer = np.zeros(self.batch_size, dtype=np.int32)
        self.target_buffer = np.zeros(self.batch_size)
        self.is_trained = False

    def _preprocess_state(self, state):
        return state.reshape(-1).astype(np.float32)

    def remember(self, state, action, reward, next_state, done):
        flat_state = self._preprocess_state(state)
        flat_next_state = self._preprocess_state(next_state)
        
        if self.is_trained:
            next_q = self.model.predict_proba([flat_next_state])[0]
            target = reward + self.gamma * np.max(next_q) * (1 - done)
        else:
            target = reward
            
        idx = len(self.action_buffer) % self.batch_size
        self.state_buffer[idx] = flat_state
        self.action_buffer[idx] = action
        self.target_buffer[idx] = target

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        flat_state = self._preprocess_state(state).reshape(1, -1)
        return self.model.predict(flat_state)[0]

    def replay(self):
        if len(self.action_buffer) < self.batch_size:
            return
            
        # Filter out invalid actions (just in case)
        valid_indices = self.action_buffer < self.action_size
        states = self.state_buffer[valid_indices]
        actions = self.action_buffer[valid_indices]
        targets = self.target_buffer[valid_indices]
        
        if len(actions) == 0:
            return
            
        with tqdm(total=1, desc="Training", leave=False) as pbar:
            self.model.fit(
                states,
                actions,
                sample_weight=np.abs(targets),
                callbacks=[lambda _: pbar.update(1)]
            )
        self.is_trained = True
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """Save model to disk"""
        joblib.dump(self.model, f"{name}.joblib")

    def load(self, name):
        """Load model from disk"""
        self.model = joblib.load(f"{name}.joblib")
        self.is_trained = True


class RuleBasedAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # lookback_window
        self.action_size = action_size  # 3
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.min_occurrences = 30  # Minimum observations to create rule
        self.min_pf = 1.2  # Minimum Profit Ratio to keep rule

        # Rule storage
        self.rule_book = {}  # {state_tuple: (action, pf, count)}
        self.memory = deque(maxlen=110000)  
        self.key_cache = {}
        self.q_table = np.zeros((2**22, 3), dtype=np.float32)  # Covers 22 features
        self.visit_counts = np.zeros((2**22, 3), dtype=np.int32)

        # Optimization tracking
        self.episode_returns = []
        self.rule_performance = defaultdict(list)

        # Feature names for debugging
        self.feature_names = [
            "Open>Close", "Close>Close1", "Close1>Close2", "Close2>Close3", "Close3>Close4", "Close4>Close5", 
            "Close>EMA9", "Close>EMA20", "Close>EMA50", "Close>EMA200",
            "Volume>MA20", "TypPrice>Mean", "RSI>50", "RSI>70", "RSI>30",
            "MACD>Signal", "ADX>25", "ADX>50", "ADX>75", "ATR>Mean", "OBV>MA20",
            "Close<BB_Lower", "Close>BB_Upper"
        ]

    def old_state_to_key(self, state):
        """Convert state to immutable tuple key"""
        return tuple(state.reshape(-1).astype(int))
        #return tuple((np.mean(state, axis=0) > 0.5).astype(int))

    def _state_to_key(self, state):
        """Cache state keys for 100x faster lookups"""
        state_bytes = state.tobytes()  # Faster than tuple conversion
        if state_bytes not in self.key_cache:
            self.key_cache[state_bytes] = tuple(state.reshape(-1).astype(int))
        return self.key_cache[state_bytes]

    def act(self, state):
        state_key = self._state_to_key(state)

        # Exploration
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)

        # Optimized exploitation
        if state_key in self.rule_book:
            action, pf, count = self.rule_book[state_key]
            # Probability weighted by rule performance
            #if random.random() < pf / (pf + 1.0):
            return action

        return 2  # Default hold

    def _state_to_index(self, state):
        """Convert binary state to integer index"""
        return sum(int(b)<<i for i,b in enumerate(state.flatten()))

    def alt_act(self, state):
        state_idx = self._state_to_index(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        return np.argmax(self.q_table[state_idx])

    def remember(self, state, action, reward, next_state, done):
        if action is not None and state is not None:
            """Store experience with volatility-based sampling"""
            self.memory.append((state, action, reward, next_state, done))

    def old_evaluate_rule(self, rewards):
        gains = sum(r for r in rewards if r > 0)
        losses = abs(sum(r for r in rewards if r < 0))
        return gains / (losses + 1e-6)  # PF > 1.2 = viable

    def evaluate_rule(self, rewards):
        rewards = np.array(rewards, dtype=np.float32)
        gains = np.sum(rewards[rewards > 0])
        losses = -np.sum(rewards[rewards < 0])
        return gains / (losses + 1e-6)

    def update_rules(self, episode_return=None):
        """Mine patterns and optimize for cumulative reward"""
        # Track episode performance
        if episode_return is not None:
            self.episode_returns.append(episode_return)
            #self.min_sharpe = max(0.3, 0.5 - (episode_return/100))

        # 1. Mine new rules
        pattern_stats = defaultdict(lambda: defaultdict(list))
        for state, action, reward, next_state, done in self.memory:
            state_key = self._state_to_key(state)
            pattern_stats[state_key][action].append(reward)

        new_rules = {}
        for pattern in pattern_stats:
            for action in pattern_stats[pattern]:
                rewards = pattern_stats[pattern][action]
                if len(rewards) < self.min_occurrences:
                    continue

                pf = self.evaluate_rule(rewards)
                if pf >= self.min_pf:
                    new_rules[pattern] = (action, pf, len(rewards))

        # 2. Update rule book with optimization
        for pattern in new_rules:
            action, pf, count = new_rules[pattern]

            # Track rule performance
            self.rule_performance[pattern].append(pf)

            # Only update if significantly better
            if pattern not in self.rule_book or pf > self.rule_book[pattern][1]:
                self.rule_book[pattern] = (action, pf, count)

        # 3. Prune underperforming rules
        to_delete = []
        for pattern in self.rule_book:
            if len(self.rule_performance[pattern]) > 5:
                trend = np.gradient(self.rule_performance[pattern])
                if np.mean(trend) < -0.1:  # Negative performance trend
                    to_delete.append(pattern)
        for pattern in to_delete:
            del self.rule_book[pattern]


        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self, name):
        """Save rules to disk"""
        joblib.dump({
            'rule_book': self.rule_book,
            'epsilon': self.epsilon,
            'performance': dict(self.rule_performance)
        }, f"{name}.joblib")

    def load(self, name):
        """Load rules from disk"""
        data = joblib.load(f"{name}.joblib")
        self.rule_book = data['rule_book']
        self.epsilon = data['epsilon']
        self.rule_performance = defaultdict(list, data.get('performance', {}))

    def print_top_rules(self, n=5):
        """Print most profitable rules"""
        sorted_rules = sorted(self.rule_book.items(),
                            key=lambda x: -x[1][1])[:n]
        print(f"\n=== Top {n} Rules (Current ε={self.epsilon:.3f}) ===")

        for pattern, (action, pf, count) in sorted_rules:
            conditions = []
            for i, val in enumerate(pattern):
                if i % len(self.feature_names) == 0:
                    conditions.append(f"\nT-{i//len(self.feature_names)}: ")
                conditions.append(f"{self.feature_names[i%len(self.feature_names)]}={val} ")

            print(f"WHEN {''.join(conditions)}")
            print(f"→ {['LONG','SHORT','DO NOTHING'][action]} "
                 f"(Profit Factor: {pf:.2f}, Samples: {count})")
            print("---")
