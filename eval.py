import gymnasium as gym
import imageio
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
model = PPO.load("ppo_halfcheetah")

obs, _ = env.reset(seed=42)
frames, total_reward = [], 0.0

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    frames.append(env.render())
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print(f"Episode reward: {total_reward:.1f}")
imageio.mimsave("trained_cheetah.mp4", frames, fps=30)