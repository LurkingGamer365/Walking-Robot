from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("HalfCheetah-v5", n_envs=4, seed=0)

model = PPO(
    "MlpPolicy",
    env,
    device="cpu"
)

model.learn(total_timesteps=1_000_000, progress_bar=True)
model.save("ppo_halfcheetah")