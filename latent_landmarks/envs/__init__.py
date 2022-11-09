import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='A1-v1',
    entry_point='latent_landmarks.envs.unitree_a1:A1Env',
    max_episode_steps=1000,
)

register(
    id='GoalA1-v1',
    entry_point='latent_landmarks.envs.create_goal_env:create_goal_env',
    max_episode_steps=100,
)