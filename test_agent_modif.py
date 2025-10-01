import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dqn_agent_modif import DQNAgent   # Pastikan class sama seperti training

# ============================================================
# 1. Environment
# ============================================================
env = gym.make("MountainCar-v0", render_mode="human")  # render animasi
state_size  = env.observation_space.shape[0]   # 2
action_size = env.action_space.n               # 3

# ============================================================
# 2. Load agent & model terlatih
# ============================================================
agent = DQNAgent(state_size, action_size)
agent.model.load_weights("dqn_mountaincar_modif.h5")
agent.epsilon = 0.01   # hampir tanpa eksplorasi, tapi tidak ditampilkan

# ============================================================
# 3. Testing
# ============================================================
episodes     = 100
max_steps    = 200
records      = []   # untuk Excel/plot

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    for t in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        state = np.reshape(next_state, [1, state_size])

        if terminated or truncated:
            # ====== Cetakan mirip training TANPA epsilon ======
            print(f"Episode: {e+1}/{episodes}, Score: {t}")

            # simpan data untuk Excel & plot
            records.append({
                "Episode": e + 1,
                "TotalReward": total_reward,
                "Steps": t        # t mulai 0, sama seperti training
            })
            break

env.close()

# ============================================================
# 4. Simpan ke Excel
# ============================================================
df = pd.DataFrame(records)
excel_filename = "mountaincar_test_results.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Hasil uji tersimpan di: {excel_filename}")

# ============================================================
# 5. Plot hasil
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(df["Episode"], df["TotalReward"], marker='o', label="Total Reward")
plt.axhline(y=-110, color='r', linestyle='--', label='Solved threshold (-110)')
plt.title("MountainCar Test Results")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
