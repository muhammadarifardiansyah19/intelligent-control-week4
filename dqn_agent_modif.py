import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    model_path = "dqn_mountaincar_shaped_model.h5"
    episodes = 1000
    batch_size = 32
    target_update_freq = 10

    if os.path.exists(model_path):
        agent.model = load_model(model_path)
        agent.target_model = clone_model(agent.model)
        agent.target_model.set_weights(agent.model.get_weights())
        print("Loaded saved model.")
        agent.epsilon = agent.epsilon_min

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(200):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward shaping: lebih besar jika posisi mobil lebih tinggi
            reward = next_state[0] + 0.5
            if next_state[0] >= 0.5:  # Goal
                reward = 100

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % target_update_freq == 0:
            agent.update_target_network()

    agent.model.save(model_path)
    print(f"Model saved to {model_path}")  
