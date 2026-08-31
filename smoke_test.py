import gymnasium as gym
import imageio

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
obs, _ = env.reset(seed=0)
frames = []
for _ in range(500):
    action = env.action_space.sample()   # random policy
    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
imageio.mimsave("random_cheetah.mp4", frames, fps=30)