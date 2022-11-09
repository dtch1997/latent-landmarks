import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='GoalA1-v1',
    entry_point='latent_landmarks.envs.create_goal_env:create_goal_env',
    max_episode_steps=100,
)