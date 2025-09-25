import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent  # pastikan sudah benar
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode='human')  # penting ini
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
agent.model.load_weights("dqn_cartpole_model_250.h5")  # atau load_model()

agent.epsilon = 0.01  # eksplorasi sangat kecil saat testing

scores = []

for e in range(20):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            scores.append(time)
            break
env.close()

# Plot hasil test
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.title('Test Episode Scores')
plt.xlabel('Episode')
plt.ylabel('Score (time steps)')
plt.grid(True)
plt.show()
